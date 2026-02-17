// gemm.cu — CUDA-cores-only GEMM (minimal but decently fast), FP16_ACCUM flag
// Includes
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>


// ----------------------------------------------------------------------------------
// Tiling configuration (safe defaults; tweak for your GPU/shape)
// ----------------------------------------------------------------------------------
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;   // try 64 if registers allow

// Threads per block: (TILE_M/4) x (TILE_N/4) = 16 x 16 = 256
constexpr int BLK_TY = TILE_M / 4;
constexpr int BLK_TX = TILE_N / 4;

// ----------------------------------------------------------------------------------
// 128-bit vectorized copy helper: global -> shared (row-major tiles of __half)
// - Copies in 16B units (8 halfs) from GLOBAL
// - Scatters as scalars into SHARED using explicit stride (handles +1 padding safely)
// s: shared tile base, shape [s_rows, s_cols_effective], leading dim s_ld (== s_cols_effective)
// g: base of global matrix (row-major), g_ld is its leading dimension.
// We only process the first s_cols "logical" columns (the effective inner length is s_ld).
// ----------------------------------------------------------------------------------
static __device__ inline void g2s_copy_half_tile_vec128(
    __half* __restrict__ s,           // shared tile base
    int s_rows,                       // number of rows in the tile
    int s_cols,                       // logical columns to fill (no padding)
    int s_ld,                         // shared leading dimension (may be s_cols+1)
    const __half* __restrict__ g,     // global base (row-major)
    int g_ld,                         // leading dimension of g
    int g_row0, int g_col0,           // global tile origin
    int M_or_K, int K_or_N,           // global matrix shape
    int tid, int nthreads)
{
    const int VEC = 8;  // 8 half = 16B per transaction
    const int total = s_rows * s_cols;

    // Bulk: 16B chunks along columns (coalesced global access)
    for (int idx = tid * VEC; idx < total; idx += nthreads * VEC) {
        int r = idx / s_cols;
        int c = idx % s_cols;

        if (c + VEC <= s_cols) {
            int gr = g_row0 + r;
            int gc = g_col0 + c;

            bool inb = (gr >= 0 && gr < M_or_K && gc >= 0 && (gc + VEC - 1) < K_or_N);
            if (inb) {
                // 16B global load (aligned because TILE_K/TILE_N are multiples of 32/64)
                const uint4* gp = reinterpret_cast<const uint4*>(&g[gr * g_ld + gc]);
                uint4 val = gp[0];

                // Scatter into shared scalars (avoid alignment assumptions in shared)
                const __half* hv = reinterpret_cast<const __half*>(&val);
                #pragma unroll
                for (int t = 0; t < VEC; ++t) {
                    s[r * s_ld + (c + t)] = hv[t];
                }
            } else {
                // Edge/tail in global
                #pragma unroll
                for (int t = 0; t < VEC; ++t) {
                    int gc_t = gc + t;
                    s[r * s_ld + (c + t)] =
                        (gr < M_or_K && gc_t < K_or_N) ? g[gr * g_ld + gc_t] : __float2half(0.f);
                }
            }
        }
    }

    // Tail if s_cols % 8 != 0
    const int remStart = (s_cols / VEC) * VEC;
    if (remStart < s_cols) {
        const int remCols = s_cols - remStart;
        for (int idx = tid; idx < s_rows * remCols; idx += nthreads) {
            int r = idx / remCols;
            int c = idx % remCols;
            int gr = g_row0 + r;
            int gc = g_col0 + remStart + c;
            s[r * s_ld + (remStart + c)] =
                (gr < M_or_K && gc < K_or_N) ? g[gr * g_ld + gc] : __float2half(0.f);
        }
    }
}

// ----------------------------------------------------------------------------------
// Main kernel templated on FP16_ACCUM:
//   - FP16_ACCUM=true  : half2 math (__hfma2), FP16 accumulation
//   - FP16_ACCUM=false : scalar FP32 FMA accumulation
// ----------------------------------------------------------------------------------
template<bool FP16_ACCUM>
__global__ void gemm_core_kernel(
    const __half* __restrict__ A, // [M, K], row-major
    const __half* __restrict__ B, // [K, N], row-major
    __half* __restrict__ C,       // [M, N], row-major
    int M, int N, int K)
{
    const int tx = threadIdx.x; // [0, BLK_TX)
    const int ty = threadIdx.y; // [0, BLK_TY)
    const int tid = ty * BLK_TX + tx; // 0..255

    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Each thread computes a 4x4 micro-tile of C
    const int row0 = block_row + ty * 4;
    const int col0 = block_col + tx * 4;

    // Padded shared memory (+1 on inner dims)
    __shared__ __half sA[2][TILE_M][TILE_K + 1];
    __shared__ __half sB[2][TILE_K][TILE_N + 1];

    // Accumulators
    __half2 acc_h2[4][2];   // two column-pairs per row (0-1, 2-3)
    float   acc_f [4][4];

    if constexpr (FP16_ACCUM) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc_h2[i][0] = __float2half2_rn(0.f);
            acc_h2[i][1] = __float2half2_rn(0.f);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc_f[i][j] = 0.f;
            }
        }
    }

    // ---------------- Preload stage 0 ----------------
    int stage = 0;
    {
        const int k0 = 0;

        // A tile: TILE_M x TILE_K starting at (block_row, k0)
        g2s_copy_half_tile_vec128(
            &sA[stage][0][0], TILE_M, TILE_K, /*s_ld=*/TILE_K + 1,
            A, /*g_ld=*/K, block_row, k0,
            /*M_or_K=*/M, /*K_or_N=*/K,
            tid, BLK_TX * BLK_TY);

        // B tile: TILE_K x TILE_N starting at (k0, block_col)
        g2s_copy_half_tile_vec128(
            &sB[stage][0][0], TILE_K, TILE_N, /*s_ld=*/TILE_N + 1,
            B, /*g_ld=*/N, k0, block_col,
            /*M_or_K=*/K, /*K_or_N=*/N,
            tid, BLK_TX * BLK_TY);
    }
    __syncthreads();

    // ---------------- Main K loop ----------------
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int next  = stage ^ 1;
        const int kNext = k0 + TILE_K;

        // Prefetch next tiles
        if (kNext < K) {
            g2s_copy_half_tile_vec128(
                &sA[next][0][0], TILE_M, TILE_K, /*s_ld=*/TILE_K + 1,
                A, K, block_row, kNext,
                M, K, tid, BLK_TX * BLK_TY);

            g2s_copy_half_tile_vec128(
                &sB[next][0][0], TILE_K, TILE_N, /*s_ld=*/TILE_N + 1,
                B, N, kNext, block_col,
                K, N, tid, BLK_TX * BLK_TY);
        }

        // Compute on current stage
        #pragma unroll 8
        for (int kk = 0; kk < TILE_K; ++kk) {
            // A fragment rows (same kk across 4 rows)
            __half a0 = (row0 + 0 < M) ? sA[stage][(row0 - block_row) + 0][kk] : __float2half(0.f);
            __half a1 = (row0 + 1 < M) ? sA[stage][(row0 - block_row) + 1][kk] : __float2half(0.f);
            __half a2 = (row0 + 2 < M) ? sA[stage][(row0 - block_row) + 2][kk] : __float2half(0.f);
            __half a3 = (row0 + 3 < M) ? sA[stage][(row0 - block_row) + 3][kk] : __float2half(0.f);

            // B fragment columns (four cols at kk)
            int bc0 = (col0 - block_col) + 0;
            int bc1 = (col0 - block_col) + 1;
            int bc2 = (col0 - block_col) + 2;
            int bc3 = (col0 - block_col) + 3;

            __half b00 = (col0 + 0 < N) ? sB[stage][kk][bc0] : __float2half(0.f);
            __half b01 = (col0 + 1 < N) ? sB[stage][kk][bc1] : __float2half(0.f);
            __half b02 = (col0 + 2 < N) ? sB[stage][kk][bc2] : __float2half(0.f);
            __half b03 = (col0 + 3 < N) ? sB[stage][kk][bc3] : __float2half(0.f);

            if constexpr (FP16_ACCUM) {
                // Pack columns in pairs
                __half2 b01_2 = __halves2half2(b00, b01);
                __half2 b23_2 = __halves2half2(b02, b03);

                // Broadcast each 'a' across 2 columns and FMA
                __half2 a0_2 = __halves2half2(a0, a0);
                __half2 a1_2 = __halves2half2(a1, a1);
                __half2 a2_2 = __halves2half2(a2, a2);
                __half2 a3_2 = __halves2half2(a3, a3);

                acc_h2[0][0] = __hfma2(a0_2, b01_2, acc_h2[0][0]);
                acc_h2[0][1] = __hfma2(a0_2, b23_2, acc_h2[0][1]);

                acc_h2[1][0] = __hfma2(a1_2, b01_2, acc_h2[1][0]);
                acc_h2[1][1] = __hfma2(a1_2, b23_2, acc_h2[1][1]);

                acc_h2[2][0] = __hfma2(a2_2, b01_2, acc_h2[2][0]);
                acc_h2[2][1] = __hfma2(a2_2, b23_2, acc_h2[2][1]);

                acc_h2[3][0] = __hfma2(a3_2, b01_2, acc_h2[3][0]);
                acc_h2[3][1] = __hfma2(a3_2, b23_2, acc_h2[3][1]);
            } else {
                float b00f = __half2float(b00), b01f = __half2float(b01);
                float b02f = __half2float(b02), b03f = __half2float(b03);
                float a0f = __half2float(a0),   a1f = __half2float(a1);
                float a2f = __half2float(a2),   a3f = __half2float(a3);

                acc_f[0][0] = __fmaf_rn(a0f, b00f, acc_f[0][0]);
                acc_f[0][1] = __fmaf_rn(a0f, b01f, acc_f[0][1]);
                acc_f[0][2] = __fmaf_rn(a0f, b02f, acc_f[0][2]);
                acc_f[0][3] = __fmaf_rn(a0f, b03f, acc_f[0][3]);

                acc_f[1][0] = __fmaf_rn(a1f, b00f, acc_f[1][0]);
                acc_f[1][1] = __fmaf_rn(a1f, b01f, acc_f[1][1]);
                acc_f[1][2] = __fmaf_rn(a1f, b02f, acc_f[1][2]);
                acc_f[1][3] = __fmaf_rn(a1f, b03f, acc_f[1][3]);

                acc_f[2][0] = __fmaf_rn(a2f, b00f, acc_f[2][0]);
                acc_f[2][1] = __fmaf_rn(a2f, b01f, acc_f[2][1]);
                acc_f[2][2] = __fmaf_rn(a2f, b02f, acc_f[2][2]);
                acc_f[2][3] = __fmaf_rn(a2f, b03f, acc_f[2][3]);

                acc_f[3][0] = __fmaf_rn(a3f, b00f, acc_f[3][0]);
                acc_f[3][1] = __fmaf_rn(a3f, b01f, acc_f[3][1]);
                acc_f[3][2] = __fmaf_rn(a3f, b02f, acc_f[3][2]);
                acc_f[3][3] = __fmaf_rn(a3f, b03f, acc_f[3][3]);
            }
        }

        __syncthreads();
        stage ^= 1;
    }

    // ---------------- Write back 4x4 micro-tile ----------------
    if constexpr (FP16_ACCUM) {
        if (row0 + 0 < M) {
            if (col0 + 0 < N) C[(row0 + 0)*N + (col0 + 0)] = __low2half (acc_h2[0][0]);
            if (col0 + 1 < N) C[(row0 + 0)*N + (col0 + 1)] = __high2half(acc_h2[0][0]);
            if (col0 + 2 < N) C[(row0 + 0)*N + (col0 + 2)] = __low2half (acc_h2[0][1]);
            if (col0 + 3 < N) C[(row0 + 0)*N + (col0 + 3)] = __high2half(acc_h2[0][1]);
        }
        if (row0 + 1 < M) {
            if (col0 + 0 < N) C[(row0 + 1)*N + (col0 + 0)] = __low2half (acc_h2[1][0]);
            if (col0 + 1 < N) C[(row0 + 1)*N + (col0 + 1)] = __high2half(acc_h2[1][0]);
            if (col0 + 2 < N) C[(row0 + 1)*N + (col0 + 2)] = __low2half (acc_h2[1][1]);
            if (col0 + 3 < N) C[(row0 + 1)*N + (col0 + 3)] = __high2half(acc_h2[1][1]);
        }
        if (row0 + 2 < M) {
            if (col0 + 0 < N) C[(row0 + 2)*N + (col0 + 0)] = __low2half (acc_h2[2][0]);
            if (col0 + 1 < N) C[(row0 + 2)*N + (col0 + 1)] = __high2half(acc_h2[2][0]);
            if (col0 + 2 < N) C[(row0 + 2)*N + (col0 + 2)] = __low2half (acc_h2[2][1]);
            if (col0 + 3 < N) C[(row0 + 2)*N + (col0 + 3)] = __high2half(acc_h2[2][1]);
        }
        if (row0 + 3 < M) {
            if (col0 + 0 < N) C[(row0 + 3)*N + (col0 + 0)] = __low2half (acc_h2[3][0]);
            if (col0 + 1 < N) C[(row0 + 3)*N + (col0 + 1)] = __high2half(acc_h2[3][0]);
            if (col0 + 2 < N) C[(row0 + 3)*N + (col0 + 2)] = __low2half (acc_h2[3][1]);
            if (col0 + 3 < N) C[(row0 + 3)*N + (col0 + 3)] = __high2half(acc_h2[3][1]);
        }
    } else {
        if (row0 + 0 < M) {
            if (col0 + 0 < N) C[(row0 + 0)*N + (col0 + 0)] = __float2half(acc_f[0][0]);
            if (col0 + 1 < N) C[(row0 + 0)*N + (col0 + 1)] = __float2half(acc_f[0][1]);
            if (col0 + 2 < N) C[(row0 + 0)*N + (col0 + 2)] = __float2half(acc_f[0][2]);
            if (col0 + 3 < N) C[(row0 + 0)*N + (col0 + 3)] = __float2half(acc_f[0][3]);
        }
        if (row0 + 1 < M) {
            if (col0 + 0 < N) C[(row0 + 1)*N + (col0 + 0)] = __float2half(acc_f[1][0]);
            if (col0 + 1 < N) C[(row0 + 1)*N + (col0 + 1)] = __float2half(acc_f[1][1]);
            if (col0 + 2 < N) C[(row0 + 1)*N + (col0 + 2)] = __float2half(acc_f[1][2]);
            if (col0 + 3 < N) C[(row0 + 1)*N + (col0 + 3)] = __float2half(acc_f[1][3]);
        }
        if (row0 + 2 < M) {
            if (col0 + 0 < N) C[(row0 + 2)*N + (col0 + 0)] = __float2half(acc_f[2][0]);
            if (col0 + 1 < N) C[(row0 + 2)*N + (col0 + 1)] = __float2half(acc_f[2][1]);
            if (col0 + 2 < N) C[(row0 + 2)*N + (col0 + 2)] = __float2half(acc_f[2][2]);
            if (col0 + 3 < N) C[(row0 + 2)*N + (col0 + 3)] = __float2half(acc_f[2][3]);
        }
        if (row0 + 3 < M) {
            if (col0 + 0 < N) C[(row0 + 3)*N + (col0 + 0)] = __float2half(acc_f[3][0]);
            if (col0 + 1 < N) C[(row0 + 3)*N + (col0 + 1)] = __float2half(acc_f[3][1]);
            if (col0 + 2 < N) C[(row0 + 3)*N + (col0 + 2)] = __float2half(acc_f[3][2]);
            if (col0 + 3 < N) C[(row0 + 3)*N + (col0 + 3)] = __float2half(acc_f[3][3]);
        }
    }
}

// === Grouped accumulation kernel: FP32 within GROUP_K, FP16 across groups =======================
template<int GROUP_K>
__global__ void gemm_core_kernel_grouped(
    const __half* __restrict__ A, // [M, K], row-major
    const __half* __restrict__ B, // [K, N], row-major
    __half* __restrict__ C,       // [M, N], row-major
    int M, int N, int K)
{
    static_assert(GROUP_K > 0, "GROUP_K must be > 0");

    const int tx   = threadIdx.x;     // [0, BLK_TX)
    const int ty   = threadIdx.y;     // [0, BLK_TY)
    const int tid  = ty * BLK_TX + tx;// 0..255

    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Thread’s 4x4 micro-tile
    const int row0 = block_row + ty * 4;
    const int col0 = block_col + tx * 4;

    // Shared (+1 padded inner dims to avoid bank conflicts)
    __shared__ __half sA[2][TILE_M][TILE_K + 1];
    __shared__ __half sB[2][TILE_K][TILE_N + 1];

    // Outer (across groups) accumulators in FP16 (packed by col pairs: 0-1, 2-3)
    __half2 acc_h2[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc_h2[i][0] = __float2half2_rn(0.f);
        acc_h2[i][1] = __float2half2_rn(0.f);
    }

    // In-group FP32 accumulators
    float gacc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) gacc[i][j] = 0.f;
    }

    auto reset_group = [&](){
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) gacc[i][j] = 0.f;
        }
    };

    auto flush_group_to_fp16 = [&](){
        // row 0
        __half2 p01 = __floats2half2_rn(gacc[0][0], gacc[0][1]);
        __half2 p23 = __floats2half2_rn(gacc[0][2], gacc[0][3]);
        acc_h2[0][0] = __hadd2(acc_h2[0][0], p01);
        acc_h2[0][1] = __hadd2(acc_h2[0][1], p23);
        // row 1
        p01 = __floats2half2_rn(gacc[1][0], gacc[1][1]);
        p23 = __floats2half2_rn(gacc[1][2], gacc[1][3]);
        acc_h2[1][0] = __hadd2(acc_h2[1][0], p01);
        acc_h2[1][1] = __hadd2(acc_h2[1][1], p23);
        // row 2
        p01 = __floats2half2_rn(gacc[2][0], gacc[2][1]);
        p23 = __floats2half2_rn(gacc[2][2], gacc[2][3]);
        acc_h2[2][0] = __hadd2(acc_h2[2][0], p01);
        acc_h2[2][1] = __hadd2(acc_h2[2][1], p23);
        // row 3
        p01 = __floats2half2_rn(gacc[3][0], gacc[3][1]);
        p23 = __floats2half2_rn(gacc[3][2], gacc[3][3]);
        acc_h2[3][0] = __hadd2(acc_h2[3][0], p01);
        acc_h2[3][1] = __hadd2(acc_h2[3][1], p23);

        reset_group();
    };

    // Fast-path masks for interior tiles (avoid per-element bounds in hot loop)
    const bool fullM = (block_row + TILE_M <= M);
    const bool fullN = (block_col + TILE_N <= N);

    // -------- Preload stage 0 --------
    int stage = 0;
    {
        const int k0 = 0;
        g2s_copy_half_tile_vec128(&sA[stage][0][0], TILE_M, TILE_K, TILE_K + 1,
                                  A, K, block_row, k0, M, K, tid, BLK_TX * BLK_TY);
        g2s_copy_half_tile_vec128(&sB[stage][0][0], TILE_K, TILE_N, TILE_N + 1,
                                  B, N, k0, block_col, K, N, tid, BLK_TX * BLK_TY);
    }
    __syncthreads();

    int gcount = 0; // group counter (kills modulo)

    // -------- Main K loop (double-buffered shared tiles) --------
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int next  = stage ^ 1;
        const int kNext = k0 + TILE_K;

        if (kNext < K) {
            g2s_copy_half_tile_vec128(&sA[next][0][0], TILE_M, TILE_K, TILE_K + 1,
                                      A, K, block_row, kNext, M, K, tid, BLK_TX * BLK_TY);
            g2s_copy_half_tile_vec128(&sB[next][0][0], TILE_K, TILE_N, TILE_N + 1,
                                      B, N, kNext, block_col, K, N, tid, BLK_TX * BLK_TY);
        }

        // --- Compute on current stage ---
        if (fullM && fullN) {
            // Interior tile: no bound checks in the hot loop
            #pragma unroll TILE_K
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int kk_global = k0 + kk;

                // A rows
                __half a0 = sA[stage][(row0 - block_row) + 0][kk];
                __half a1 = sA[stage][(row0 - block_row) + 1][kk];
                __half a2 = sA[stage][(row0 - block_row) + 2][kk];
                __half a3 = sA[stage][(row0 - block_row) + 3][kk];

                // B cols
                const int bc0 = (col0 - block_col) + 0;
                const int bc1 = bc0 + 1;
                const int bc2 = bc0 + 2;
                const int bc3 = bc0 + 3;

                __half b00 = sB[stage][kk][bc0];
                __half b01 = sB[stage][kk][bc1];
                __half b02 = sB[stage][kk][bc2];
                __half b03 = sB[stage][kk][bc3];

                // Convert (A scalars, B as pairs)
                float a0f = __half2float(a0), a1f = __half2float(a1);
                float a2f = __half2float(a2), a3f = __half2float(a3);

                __half2 b01_2 = __halves2half2(b00, b01);
                __half2 b23_2 = __halves2half2(b02, b03);
                float2 b01f2  = __half22float2(b01_2); // x=b00f, y=b01f
                float2 b23f2  = __half22float2(b23_2); // x=b02f, y=b03f

                // FP32 FMA within the group
                gacc[0][0] = __fmaf_rn(a0f, b01f2.x, gacc[0][0]);
                gacc[0][1] = __fmaf_rn(a0f, b01f2.y, gacc[0][1]);
                gacc[0][2] = __fmaf_rn(a0f, b23f2.x, gacc[0][2]);
                gacc[0][3] = __fmaf_rn(a0f, b23f2.y, gacc[0][3]);

                gacc[1][0] = __fmaf_rn(a1f, b01f2.x, gacc[1][0]);
                gacc[1][1] = __fmaf_rn(a1f, b01f2.y, gacc[1][1]);
                gacc[1][2] = __fmaf_rn(a1f, b23f2.x, gacc[1][2]);
                gacc[1][3] = __fmaf_rn(a1f, b23f2.y, gacc[1][3]);

                gacc[2][0] = __fmaf_rn(a2f, b01f2.x, gacc[2][0]);
                gacc[2][1] = __fmaf_rn(a2f, b01f2.y, gacc[2][1]);
                gacc[2][2] = __fmaf_rn(a2f, b23f2.x, gacc[2][2]);
                gacc[2][3] = __fmaf_rn(a2f, b23f2.y, gacc[2][3]);

                gacc[3][0] = __fmaf_rn(a3f, b01f2.x, gacc[3][0]);
                gacc[3][1] = __fmaf_rn(a3f, b01f2.y, gacc[3][1]);
                gacc[3][2] = __fmaf_rn(a3f, b23f2.x, gacc[3][2]);
                gacc[3][3] = __fmaf_rn(a3f, b23f2.y, gacc[3][3]);

                // Group boundary?
                if (++gcount == GROUP_K || kk_global + 1 == K) {
                    flush_group_to_fp16();
                    gcount = 0;
                }
            }
        } else {
            // Edge tile: keep bounds
            #pragma unroll TILE_K
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int kk_global = k0 + kk;

                __half a0 = (row0 + 0 < M) ? sA[stage][(row0 - block_row) + 0][kk] : __float2half(0.f);
                __half a1 = (row0 + 1 < M) ? sA[stage][(row0 - block_row) + 1][kk] : __float2half(0.f);
                __half a2 = (row0 + 2 < M) ? sA[stage][(row0 - block_row) + 2][kk] : __float2half(0.f);
                __half a3 = (row0 + 3 < M) ? sA[stage][(row0 - block_row) + 3][kk] : __float2half(0.f);

                const int bc0 = (col0 - block_col) + 0;
                const int bc1 = bc0 + 1;
                const int bc2 = bc0 + 2;
                const int bc3 = bc0 + 3;

                __half b00 = (col0 + 0 < N) ? sB[stage][kk][bc0] : __float2half(0.f);
                __half b01 = (col0 + 1 < N) ? sB[stage][kk][bc1] : __float2half(0.f);
                __half b02 = (col0 + 2 < N) ? sB[stage][kk][bc2] : __float2half(0.f);
                __half b03 = (col0 + 3 < N) ? sB[stage][kk][bc3] : __float2half(0.f);

                float a0f = __half2float(a0), a1f = __half2float(a1);
                float a2f = __half2float(a2), a3f = __half2float(a3);

                __half2 b01_2 = __halves2half2(b00, b01);
                __half2 b23_2 = __halves2half2(b02, b03);
                float2 b01f2  = __half22float2(b01_2);
                float2 b23f2  = __half22float2(b23_2);

                gacc[0][0] = __fmaf_rn(a0f, b01f2.x, gacc[0][0]);
                gacc[0][1] = __fmaf_rn(a0f, b01f2.y, gacc[0][1]);
                gacc[0][2] = __fmaf_rn(a0f, b23f2.x, gacc[0][2]);
                gacc[0][3] = __fmaf_rn(a0f, b23f2.y, gacc[0][3]);

                gacc[1][0] = __fmaf_rn(a1f, b01f2.x, gacc[1][0]);
                gacc[1][1] = __fmaf_rn(a1f, b01f2.y, gacc[1][1]);
                gacc[1][2] = __fmaf_rn(a1f, b23f2.x, gacc[1][2]);
                gacc[1][3] = __fmaf_rn(a1f, b23f2.y, gacc[1][3]);

                gacc[2][0] = __fmaf_rn(a2f, b01f2.x, gacc[2][0]);
                gacc[2][1] = __fmaf_rn(a2f, b01f2.y, gacc[2][1]);
                gacc[2][2] = __fmaf_rn(a2f, b23f2.x, gacc[2][2]);
                gacc[2][3] = __fmaf_rn(a2f, b23f2.y, gacc[2][3]);

                gacc[3][0] = __fmaf_rn(a3f, b01f2.x, gacc[3][0]);
                gacc[3][1] = __fmaf_rn(a3f, b01f2.y, gacc[3][1]);
                gacc[3][2] = __fmaf_rn(a3f, b23f2.x, gacc[3][2]);
                gacc[3][3] = __fmaf_rn(a3f, b23f2.y, gacc[3][3]);

                if (++gcount == GROUP_K || kk_global + 1 == K) {
                    flush_group_to_fp16();
                    gcount = 0;
                }
            }
        }

        __syncthreads();
        stage ^= 1;
    }

    // Write back 4x4 from acc_h2
    if (row0 + 0 < M) {
        if (col0 + 0 < N) C[(row0 + 0)*N + (col0 + 0)] = __low2half (acc_h2[0][0]);
        if (col0 + 1 < N) C[(row0 + 0)*N + (col0 + 1)] = __high2half(acc_h2[0][0]);
        if (col0 + 2 < N) C[(row0 + 0)*N + (col0 + 2)] = __low2half (acc_h2[0][1]);
        if (col0 + 3 < N) C[(row0 + 0)*N + (col0 + 3)] = __high2half(acc_h2[0][1]);
    }
    if (row0 + 1 < M) {
        if (col0 + 0 < N) C[(row0 + 1)*N + (col0 + 0)] = __low2half (acc_h2[1][0]);
        if (col0 + 1 < N) C[(row0 + 1)*N + (col0 + 1)] = __high2half(acc_h2[1][0]);
        if (col0 + 2 < N) C[(row0 + 1)*N + (col0 + 2)] = __low2half (acc_h2[1][1]);
        if (col0 + 3 < N) C[(row0 + 1)*N + (col0 + 3)] = __high2half(acc_h2[1][1]);
    }
    if (row0 + 2 < M) {
        if (col0 + 0 < N) C[(row0 + 2)*N + (col0 + 0)] = __low2half (acc_h2[2][0]);
        if (col0 + 1 < N) C[(row0 + 2)*N + (col0 + 1)] = __high2half(acc_h2[2][0]);
        if (col0 + 2 < N) C[(row0 + 2)*N + (col0 + 2)] = __low2half (acc_h2[2][1]);
        if (col0 + 3 < N) C[(row0 + 2)*N + (col0 + 3)] = __high2half(acc_h2[2][1]);
    }
    if (row0 + 3 < M) {
        if (col0 + 0 < N) C[(row0 + 3)*N + (col0 + 0)] = __low2half (acc_h2[3][0]);
        if (col0 + 1 < N) C[(row0 + 3)*N + (col0 + 1)] = __high2half(acc_h2[3][0]);
        if (col0 + 2 < N) C[(row0 + 3)*N + (col0 + 2)] = __low2half (acc_h2[3][1]);
        if (col0 + 3 < N) C[(row0 + 3)*N + (col0 + 3)] = __high2half(acc_h2[3][1]);
    }
}

// ----------------------------------------------------------------------------------
// C++ wrappers
// ----------------------------------------------------------------------------------
static inline void check_args(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf && C.scalar_type() == at::kHalf,
                "This impl expects half inputs/outputs");
    TORCH_CHECK(A.dim()==2 && B.dim()==2 && C.dim()==2, "2D tensors expected");
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch");
    TORCH_CHECK(A.device() == B.device() && A.device() == C.device(), "A, B, and C must be on the same CUDA device");
    TORCH_CHECK(A.size(0) == C.size(0) && B.size(1) == C.size(1), "Output shape mismatch");
}

void gemm_fp16_accum_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    // FP16_ACCUM = true
    check_args(A, B, C);

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto Cc = C.contiguous();

    const int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);

    dim3 block(BLK_TX, BLK_TY); // 16 x 16 = 256 threads
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_core_kernel</*FP16_ACCUM=*/true><<<grid, block>>>(
        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
        M, N, K);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_fp16_accum kernel failed: ", cudaGetErrorString(err));

    if (!C.is_contiguous()) C.copy_(Cc);
}

void gemm_fp32_accum_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    // FP16_ACCUM = false
    check_args(A, B, C);

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto Cc = C.contiguous();

    const int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);

    dim3 block(BLK_TX, BLK_TY); // 16 x 16 = 256 threads
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_core_kernel</*FP16_ACCUM=*/false><<<grid, block>>>(
        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
        M, N, K);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_fp32_accum kernel failed: ", cudaGetErrorString(err));

    if (!C.is_contiguous()) C.copy_(Cc);
}

void gemm_fp16_grouped_accum_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int group_k) {
    check_args(A, B, C);
    TORCH_CHECK(group_k == 16 || group_k == 32 || group_k == 64 || group_k == 128,
                "group_k must be one of {16,32,64,128}");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto Cc = C.contiguous();

    const int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);

    dim3 block(BLK_TX, BLK_TY); // 16 x 16
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    switch (group_k) {
        case 16:
            gemm_core_kernel_grouped<16><<<grid, block>>>(
                reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                M, N, K);
            break;
        case 32:
            gemm_core_kernel_grouped<32><<<grid, block>>>(
                reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                M, N, K);
            break;
        case 64:
            gemm_core_kernel_grouped<64><<<grid, block>>>(
                reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                M, N, K);
            break;
        case 128:
            gemm_core_kernel_grouped<128><<<grid, block>>>(
                reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                M, N, K);
            break;
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_fp16_grouped_accum kernel failed: ", cudaGetErrorString(err));

    if (!C.is_contiguous()) C.copy_(Cc);
}

// ==================================================================================
// E3M4 (MX-style) fake-quant helpers (nearest-even), bias=3, normals e in [-2,3]
// Min subnormal: 2^-6 = 0.015625, Min normal: 2^-2 = 0.25, Max normal: 15.5
// We quantize-dequantize a float using a power-of-two scale:
//   scale = 2^floor(log2(max_abs) - 3)
// so the scaled max lies in [8, 16). We then round-to-nearest-even and clamp
// to the supported E3M4 range (max 15.5) during quant-dequant.
// ==================================================================================
static __device__ inline float e3m4_quantize_dequant(float x, float scale) {
    // Preserve sign; operate on magnitude
    float sgn = copysignf(1.0f, x);
    float a = fabsf(x);
    if (a == 0.0f) return copysignf(0.0f, x); // preserve signed zero

    const float MIN_NORMAL = 0.25f;     // 2^-2
    const float MIN_SUB    = 0.015625f; // 2^-6
    const float MAX_NORMAL = 15.5f;     // (1+15/16)*2^3

    float y = a / scale; // scaled magnitude
    // Explicit policy for non-finites
    if (!isfinite(y)) {
        if (isnan(y)) return copysignf(NAN, x);
        // +/- Inf -> clamp to max normal after dequant
        return sgn * MAX_NORMAL * scale;
    }

    // Values below half of the smallest subnormal will round to zero
    if (y < 0.5f * MIN_SUB) {
        return 0.0f;
    }

    // Normal numbers path
    if (y >= MIN_NORMAL) {
        // e = floor(log2(y)), clamp to [-2,3]
        int e = (int)floorf(log2f(y));
        if (e < -2) e = -2;
        if (e >  3) e =  3; // should not happen if scale chosen as specified

        float pow2e = ldexpf(1.0f, e); // 2^e
        float frac = y / pow2e - 1.0f; // in [0,1)
        float mant_f = nearbyintf(frac * 16.0f); // nearest-even to 4-bit mantissa

        // Handle potential carry from rounding (rare if scale correct)
        if (mant_f >= 16.0f) {
            mant_f = 0.0f;
            ++e;
            if (e > 3) { // clamp to max normal
                e = 3;
                mant_f = 15.0f;
            }
            pow2e = ldexpf(1.0f, e);
        }

        float yq = (1.0f + mant_f * (1.0f / 16.0f)) * pow2e;
        if (yq > MAX_NORMAL) yq = MAX_NORMAL; // hard safety clamp
        return sgn * yq * scale;
    }

    // Subnormal path: value = (mant/16) * 2^-2
    float mant_f = nearbyintf(y * 64.0f); // y * 2^6
    if (mant_f <= 0.0f) return 0.0f;
    if (mant_f >= 16.0f) {
        // Promote to min normal (0.25)
        float yq = MIN_NORMAL;
        return sgn * yq * scale;
    }
    float yq = (mant_f * (1.0f / 16.0f)) * MIN_NORMAL; // mant/16 * 2^-2
    return sgn * yq * scale;
}

static __device__ inline float compute_scale_pow2(float max_abs) {
    // MX FP8 scale policy: scale = 2^floor(log2(max_abs) - 3)
    // Guard zero/negative
    if (max_abs <= 0.0f) return 1.0f;
    int exp2 = ilogbf(max_abs);           // floor(log2(max_abs))
    return ldexpf(1.0f, exp2 - 3);        // 2^(exp2-3)
}

// ==================================================================================
// BlockQuant kernel: FP32 accumulate within GROUP_K, then per-row GROUP_N fake-quant
// to E3M4 with power-of-two scale, dequantize, then accumulate across groups in FP16
// (global per-thread accumulator). Final write stores FP16.
// ==================================================================================
template<int GROUP_K, int GROUP_N>
__global__ void gemm_blockquant_mxfp8_core_kernel(
    const __half* __restrict__ A, // [M, K], row-major
    const __half* __restrict__ B, // [K, N], row-major
    __half* __restrict__ C,       // [M, N], row-major
    int M, int N, int K)
{
    const int tx = threadIdx.x; // [0, BLK_TX)
    const int ty = threadIdx.y; // [0, BLK_TY)
    const int tid = ty * BLK_TX + tx; // 0..255

    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Thread’s 4x4 micro-tile origin in C
    const int row0 = block_row + ty * 4;
    const int col0 = block_col + tx * 4;

    // Shared tiles (+1 padding to avoid bank conflicts)
    __shared__ __half sA[2][TILE_M][TILE_K + 1];
    __shared__ __half sB[2][TILE_K][TILE_N + 1];

    // Shared staging for quantization at GROUP_K boundaries (FP32)
    // Pad +1 on inner dim to reduce bank conflicts
    __shared__ float sTile[TILE_M][TILE_N + 1];

    // Global (across all K) FP16 accumulator per-thread for its 4x4 outputs
    __half gacc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) gacc[i][j] = __float2half(0.f);
    }

    // Intra-GROUP_K FP32 partials
    float pacc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) pacc[i][j] = 0.f;
    }

    const bool fullM = (block_row + TILE_M <= M);
    const bool fullN = (block_col + TILE_N <= N);

    // Preload stage 0
    int stage = 0;
    {
        const int k0 = 0;
        g2s_copy_half_tile_vec128(&sA[stage][0][0], TILE_M, TILE_K, TILE_K + 1,
                                  A, K, block_row, k0, M, K, tid, BLK_TX * BLK_TY);
        g2s_copy_half_tile_vec128(&sB[stage][0][0], TILE_K, TILE_N, TILE_N + 1,
                                  B, N, k0, block_col, K, N, tid, BLK_TX * BLK_TY);
    }
    __syncthreads();

    int processed_in_group = 0; // counts kk_global within current GROUP_K

    auto flush_group = [&]() {
        // Stage pacc into shared tile
        // Determine tile extents for edge blocks
        const int tileM = min(TILE_M, M - block_row);
        const int tileN = min(TILE_N, N - block_col);

        // Write this thread's 4x4 pacc into sTile
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = (row0 - block_row) + i;
            if (r >= tileM) break;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = (col0 - block_col) + j;
                if (c < tileN) sTile[r][c] = pacc[i][j];
            }
        }

        __syncthreads();

        // Per-row, per-GROUP_N segment quantization across the entire tile width
        // Use ceil-div for robustness if TILE_N is not divisible by GROUP_N
        const int groups_per_row = (tileN + GROUP_N - 1) / GROUP_N; // ceil-div
        const int total_groups   = tileM * groups_per_row;

        for (int g = tid; g < total_groups; g += BLK_TX * BLK_TY) {
            int r  = g / groups_per_row;
            int gi = g % groups_per_row;
            int c0 = gi * GROUP_N;

            // Compute max abs in this group (guard right edge)
            float max_abs = 0.f;
            #pragma unroll
            for (int dc = 0; dc < GROUP_N; ++dc) {
                int cc = c0 + dc;
                if (cc >= tileN) break;
                float av = fabsf(sTile[r][cc]);
                if (av > max_abs) max_abs = av;
            }
            float scale = compute_scale_pow2(max_abs);

            // Quantize-dequantize in-place (guard right edge)
            #pragma unroll
            for (int dc = 0; dc < GROUP_N; ++dc) {
                int cc = c0 + dc;
                if (cc >= tileN) break;
                float v = sTile[r][cc];
                sTile[r][cc] = e3m4_quantize_dequant(v, scale);
            }
        }

        __syncthreads();

        // Accumulate quantized values from sTile into global FP16 accum registers
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = (row0 - block_row) + i;
            if (r >= tileM) break;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = (col0 - block_col) + j;
                if (c < tileN) {
                    __half v = __float2half(sTile[r][c]);
                    gacc[i][j] = __hadd(gacc[i][j], v);
                }
            }
        }

        // Reset intra-group partials
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) pacc[i][j] = 0.f;
        }

        processed_in_group = 0;
        __syncthreads();
    };

    // Main K loop (double-buffered)
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int next  = stage ^ 1;
        const int kNext = k0 + TILE_K;

        if (kNext < K) {
            g2s_copy_half_tile_vec128(&sA[next][0][0], TILE_M, TILE_K, TILE_K + 1,
                                      A, K, block_row, kNext, M, K, tid, BLK_TX * BLK_TY);
            g2s_copy_half_tile_vec128(&sB[next][0][0], TILE_K, TILE_N, TILE_N + 1,
                                      B, N, kNext, block_col, K, N, tid, BLK_TX * BLK_TY);
        }

        // Compute on current stage
        if (fullM && fullN) {
            #pragma unroll TILE_K
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int kk_global = k0 + kk;

                // A rows
                __half a0 = sA[stage][(row0 - block_row) + 0][kk];
                __half a1 = sA[stage][(row0 - block_row) + 1][kk];
                __half a2 = sA[stage][(row0 - block_row) + 2][kk];
                __half a3 = sA[stage][(row0 - block_row) + 3][kk];

                // B cols
                const int bc0 = (col0 - block_col) + 0;
                const int bc1 = bc0 + 1;
                const int bc2 = bc0 + 2;
                const int bc3 = bc0 + 3;

                __half b00 = sB[stage][kk][bc0];
                __half b01 = sB[stage][kk][bc1];
                __half b02 = sB[stage][kk][bc2];
                __half b03 = sB[stage][kk][bc3];

                // Convert to float and accumulate into intra-group partials
                float b00f = __half2float(b00), b01f = __half2float(b01);
                float b02f = __half2float(b02), b03f = __half2float(b03);
                float a0f = __half2float(a0),   a1f = __half2float(a1);
                float a2f = __half2float(a2),   a3f = __half2float(a3);

                pacc[0][0] = __fmaf_rn(a0f, b00f, pacc[0][0]);
                pacc[0][1] = __fmaf_rn(a0f, b01f, pacc[0][1]);
                pacc[0][2] = __fmaf_rn(a0f, b02f, pacc[0][2]);
                pacc[0][3] = __fmaf_rn(a0f, b03f, pacc[0][3]);

                pacc[1][0] = __fmaf_rn(a1f, b00f, pacc[1][0]);
                pacc[1][1] = __fmaf_rn(a1f, b01f, pacc[1][1]);
                pacc[1][2] = __fmaf_rn(a1f, b02f, pacc[1][2]);
                pacc[1][3] = __fmaf_rn(a1f, b03f, pacc[1][3]);

                pacc[2][0] = __fmaf_rn(a2f, b00f, pacc[2][0]);
                pacc[2][1] = __fmaf_rn(a2f, b01f, pacc[2][1]);
                pacc[2][2] = __fmaf_rn(a2f, b02f, pacc[2][2]);
                pacc[2][3] = __fmaf_rn(a2f, b03f, pacc[2][3]);

                pacc[3][0] = __fmaf_rn(a3f, b00f, pacc[3][0]);
                pacc[3][1] = __fmaf_rn(a3f, b01f, pacc[3][1]);
                pacc[3][2] = __fmaf_rn(a3f, b02f, pacc[3][2]);
                pacc[3][3] = __fmaf_rn(a3f, b03f, pacc[3][3]);

                if (++processed_in_group == GROUP_K || kk_global + 1 == K) {
                    flush_group();
                }
            }
        } else {
            // Edge tile with bounds checks
            #pragma unroll TILE_K
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int kk_global = k0 + kk;

                __half a0 = (row0 + 0 < M) ? sA[stage][(row0 - block_row) + 0][kk] : __float2half(0.f);
                __half a1 = (row0 + 1 < M) ? sA[stage][(row0 - block_row) + 1][kk] : __float2half(0.f);
                __half a2 = (row0 + 2 < M) ? sA[stage][(row0 - block_row) + 2][kk] : __float2half(0.f);
                __half a3 = (row0 + 3 < M) ? sA[stage][(row0 - block_row) + 3][kk] : __float2half(0.f);

                const int bc0 = (col0 - block_col) + 0;
                const int bc1 = bc0 + 1;
                const int bc2 = bc0 + 2;
                const int bc3 = bc0 + 3;

                __half b00 = (col0 + 0 < N) ? sB[stage][kk][bc0] : __float2half(0.f);
                __half b01 = (col0 + 1 < N) ? sB[stage][kk][bc1] : __float2half(0.f);
                __half b02 = (col0 + 2 < N) ? sB[stage][kk][bc2] : __float2half(0.f);
                __half b03 = (col0 + 3 < N) ? sB[stage][kk][bc3] : __float2half(0.f);

                float b00f = __half2float(b00), b01f = __half2float(b01);
                float b02f = __half2float(b02), b03f = __half2float(b03);
                float a0f = __half2float(a0),   a1f = __half2float(a1);
                float a2f = __half2float(a2),   a3f = __half2float(a3);

                pacc[0][0] = __fmaf_rn(a0f, b00f, pacc[0][0]);
                pacc[0][1] = __fmaf_rn(a0f, b01f, pacc[0][1]);
                pacc[0][2] = __fmaf_rn(a0f, b02f, pacc[0][2]);
                pacc[0][3] = __fmaf_rn(a0f, b03f, pacc[0][3]);

                pacc[1][0] = __fmaf_rn(a1f, b00f, pacc[1][0]);
                pacc[1][1] = __fmaf_rn(a1f, b01f, pacc[1][1]);
                pacc[1][2] = __fmaf_rn(a1f, b02f, pacc[1][2]);
                pacc[1][3] = __fmaf_rn(a1f, b03f, pacc[1][3]);

                pacc[2][0] = __fmaf_rn(a2f, b00f, pacc[2][0]);
                pacc[2][1] = __fmaf_rn(a2f, b01f, pacc[2][1]);
                pacc[2][2] = __fmaf_rn(a2f, b02f, pacc[2][2]);
                pacc[2][3] = __fmaf_rn(a2f, b03f, pacc[2][3]);

                pacc[3][0] = __fmaf_rn(a3f, b00f, pacc[3][0]);
                pacc[3][1] = __fmaf_rn(a3f, b01f, pacc[3][1]);
                pacc[3][2] = __fmaf_rn(a3f, b02f, pacc[3][2]);
                pacc[3][3] = __fmaf_rn(a3f, b03f, pacc[3][3]);

                if (++processed_in_group == GROUP_K || kk_global + 1 == K) {
                    flush_group();
                }
            }
        }

        __syncthreads();
        stage ^= 1;
    }

    // Final write: convert gacc to fp16 and store 4x4 micro-tile
    if (row0 + 0 < M) {
        if (col0 + 0 < N) C[(row0 + 0)*N + (col0 + 0)] = gacc[0][0];
        if (col0 + 1 < N) C[(row0 + 0)*N + (col0 + 1)] = gacc[0][1];
        if (col0 + 2 < N) C[(row0 + 0)*N + (col0 + 2)] = gacc[0][2];
        if (col0 + 3 < N) C[(row0 + 0)*N + (col0 + 3)] = gacc[0][3];
    }
    if (row0 + 1 < M) {
        if (col0 + 0 < N) C[(row0 + 1)*N + (col0 + 0)] = gacc[1][0];
        if (col0 + 1 < N) C[(row0 + 1)*N + (col0 + 1)] = gacc[1][1];
        if (col0 + 2 < N) C[(row0 + 1)*N + (col0 + 2)] = gacc[1][2];
        if (col0 + 3 < N) C[(row0 + 1)*N + (col0 + 3)] = gacc[1][3];
    }
    if (row0 + 2 < M) {
        if (col0 + 0 < N) C[(row0 + 2)*N + (col0 + 0)] = gacc[2][0];
        if (col0 + 1 < N) C[(row0 + 2)*N + (col0 + 1)] = gacc[2][1];
        if (col0 + 2 < N) C[(row0 + 2)*N + (col0 + 2)] = gacc[2][2];
        if (col0 + 3 < N) C[(row0 + 2)*N + (col0 + 3)] = gacc[2][3];
    }
    if (row0 + 3 < M) {
        if (col0 + 0 < N) C[(row0 + 3)*N + (col0 + 0)] = gacc[3][0];
        if (col0 + 1 < N) C[(row0 + 3)*N + (col0 + 1)] = gacc[3][1];
        if (col0 + 2 < N) C[(row0 + 3)*N + (col0 + 2)] = gacc[3][2];
        if (col0 + 3 < N) C[(row0 + 3)*N + (col0 + 3)] = gacc[3][3];
    }
}

// ----------------------------------------------------------------------------------
// C++ wrapper for BlockQuant
// ----------------------------------------------------------------------------------
void gemm_blockquant_mxfp8_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int group_k, int group_n) {
    check_args(A, B, C);
    TORCH_CHECK(group_k == 64 || group_k == 128 || group_k == 256,
                "group_k must be one of {64,128,256}");
    TORCH_CHECK(group_n == 4 || group_n == 8 || group_n == 16 || group_n == 32,
                "group_n must be one of {4,8,16,32}");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto Cc = C.contiguous();

    const int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);

    TORCH_CHECK((K % group_k) == 0, "K must be divisible by group_k");
    TORCH_CHECK((N % group_n) == 0, "N must be divisible by group_n");

    dim3 block(BLK_TX, BLK_TY); // 16 x 16 = 256
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    // Dispatch on templates
    if (group_k == 64) {
        switch (group_n) {
            case 4:  gemm_blockquant_mxfp8_core_kernel<64,4><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 8:  gemm_blockquant_mxfp8_core_kernel<64,8><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 16: gemm_blockquant_mxfp8_core_kernel<64,16><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 32: gemm_blockquant_mxfp8_core_kernel<64,32><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
        }
    } else if (group_k == 128) {
        switch (group_n) {
            case 4:  gemm_blockquant_mxfp8_core_kernel<128,4><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 8:  gemm_blockquant_mxfp8_core_kernel<128,8><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 16: gemm_blockquant_mxfp8_core_kernel<128,16><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 32: gemm_blockquant_mxfp8_core_kernel<128,32><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
        }
    } else { // 256
        switch (group_n) {
            case 4:  gemm_blockquant_mxfp8_core_kernel<256,4><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 8:  gemm_blockquant_mxfp8_core_kernel<256,8><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 16: gemm_blockquant_mxfp8_core_kernel<256,16><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
            case 32: gemm_blockquant_mxfp8_core_kernel<256,32><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()),
                        M, N, K); break;
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_blockquant_mxfp8 kernel failed: ", cudaGetErrorString(err));

    if (!C.is_contiguous()) C.copy_(Cc);
}

// ==================================================================================
// INT8 (power-of-two scaled) BlockQuant kernel
// Same flow as FP8 E3M4 variant above, but we quantize to signed int8 range
// [-127,127] with a per-(row,GROUP_N) power-of-two scale selected so that
// max_abs/scale <= 127. Then we dequantize (q * scale) and accumulate.
// This simulates an MX-style int8 block quantization.
// ==================================================================================

static __device__ inline float compute_scale_pow2_int8(float max_abs) {
    // MX INT8 scale policy: scale = 2^floor(log2(max_abs) - 6)
    // This ensures scaled max in [64,128). We then round-to-nearest-even and
    // clip to [-127,127] during quant-dequant.
    if (max_abs <= 0.f) return 1.f;
    int exp2 = ilogbf(max_abs);          // floor(log2(max_abs))
    return ldexpf(1.f, exp2 - 6);        // 2^(exp2-6)
}

template<int GROUP_K, int GROUP_N>
__global__ void gemm_blockquant_mxint8_core_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLK_TX + tx;

    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    const int row0 = block_row + ty * 4;
    const int col0 = block_col + tx * 4;

    __shared__ __half sA[2][TILE_M][TILE_K + 1];
    __shared__ __half sB[2][TILE_K][TILE_N + 1];
    __shared__ float  sTile[TILE_M][TILE_N + 1];

    __half gacc[4][4];
    float  pacc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) { gacc[i][j] = __float2half(0.f); pacc[i][j] = 0.f; }
    }

    const bool fullM = (block_row + TILE_M <= M);
    const bool fullN = (block_col + TILE_N <= N);

    int stage = 0;
    {
        const int k0 = 0;
        g2s_copy_half_tile_vec128(&sA[stage][0][0], TILE_M, TILE_K, TILE_K + 1,
                                  A, K, block_row, k0, M, K, tid, BLK_TX * BLK_TY);
        g2s_copy_half_tile_vec128(&sB[stage][0][0], TILE_K, TILE_N, TILE_N + 1,
                                  B, N, k0, block_col, K, N, tid, BLK_TX * BLK_TY);
    }
    __syncthreads();

    int processed_in_group = 0;

    auto flush_group = [&]() {
        const int tileM = min(TILE_M, M - block_row);
        const int tileN = min(TILE_N, N - block_col);

        // Stage pacc into shared
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = (row0 - block_row) + i;
            if (r >= tileM) break;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = (col0 - block_col) + j;
                if (c < tileN) sTile[r][c] = pacc[i][j];
            }
        }
        __syncthreads();

        const int groups_per_row = (tileN + GROUP_N - 1) / GROUP_N;
        const int total_groups = tileM * groups_per_row;

        for (int g = tid; g < total_groups; g += BLK_TX * BLK_TY) {
            int r = g / groups_per_row;
            int gi = g % groups_per_row;
            int c0 = gi * GROUP_N;

            float max_abs = 0.f;
            #pragma unroll
            for (int dc = 0; dc < GROUP_N; ++dc) {
                int cc = c0 + dc;
                if (cc >= tileN) break;
                float av = fabsf(sTile[r][cc]);
                if (av > max_abs) max_abs = av;
            }
            float scale = compute_scale_pow2_int8(max_abs);
            const float inv_scale = 1.f / scale;

            // Quantize-dequantize with int8
            #pragma unroll
            for (int dc = 0; dc < GROUP_N; ++dc) {
                int cc = c0 + dc;
                if (cc >= tileN) break;
                float v = sTile[r][cc] * inv_scale; // scaled
                // Handle non-finites explicitly
                if (!isfinite(v)) {
                    if (isnan(v)) { sTile[r][cc] = NAN; continue; }
                    // inf -> clamp
                    v = copysignf(127.f, v);
                }
                float q = nearbyintf(v); // nearest-even
                if (q > 127.f) q = 127.f;
                if (q < -127.f) q = -127.f; // symmetric range
                sTile[r][cc] = q * scale; // dequantized value
            }
        }
        __syncthreads();

        // Accumulate (global FP16)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = (row0 - block_row) + i;
            if (r >= tileM) break;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c = (col0 - block_col) + j;
                if (c < tileN) {
                    __half v = __float2half(sTile[r][c]);
                    gacc[i][j] = __hadd(gacc[i][j], v);
                }
            }
        }

        // Reset pacc
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) pacc[i][j] = 0.f;
        }
        processed_in_group = 0;
        __syncthreads();
    };

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int next = stage ^ 1;
        const int kNext = k0 + TILE_K;
        if (kNext < K) {
            g2s_copy_half_tile_vec128(&sA[next][0][0], TILE_M, TILE_K, TILE_K + 1,
                                      A, K, block_row, kNext, M, K, tid, BLK_TX * BLK_TY);
            g2s_copy_half_tile_vec128(&sB[next][0][0], TILE_K, TILE_N, TILE_N + 1,
                                      B, N, kNext, block_col, K, N, tid, BLK_TX * BLK_TY);
        }

        if (fullM && fullN) {
            #pragma unroll TILE_K
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int kk_global = k0 + kk;
                __half a0 = sA[stage][(row0 - block_row) + 0][kk];
                __half a1 = sA[stage][(row0 - block_row) + 1][kk];
                __half a2 = sA[stage][(row0 - block_row) + 2][kk];
                __half a3 = sA[stage][(row0 - block_row) + 3][kk];
                const int bc0 = (col0 - block_col) + 0;
                const int bc1 = bc0 + 1;
                const int bc2 = bc0 + 2;
                const int bc3 = bc0 + 3;
                __half b00 = sB[stage][kk][bc0];
                __half b01 = sB[stage][kk][bc1];
                __half b02 = sB[stage][kk][bc2];
                __half b03 = sB[stage][kk][bc3];
                float a0f = __half2float(a0), a1f = __half2float(a1);
                float a2f = __half2float(a2), a3f = __half2float(a3);
                float b00f = __half2float(b00), b01f = __half2float(b01);
                float b02f = __half2float(b02), b03f = __half2float(b03);
                pacc[0][0] = __fmaf_rn(a0f, b00f, pacc[0][0]);
                pacc[0][1] = __fmaf_rn(a0f, b01f, pacc[0][1]);
                pacc[0][2] = __fmaf_rn(a0f, b02f, pacc[0][2]);
                pacc[0][3] = __fmaf_rn(a0f, b03f, pacc[0][3]);
                pacc[1][0] = __fmaf_rn(a1f, b00f, pacc[1][0]);
                pacc[1][1] = __fmaf_rn(a1f, b01f, pacc[1][1]);
                pacc[1][2] = __fmaf_rn(a1f, b02f, pacc[1][2]);
                pacc[1][3] = __fmaf_rn(a1f, b03f, pacc[1][3]);
                pacc[2][0] = __fmaf_rn(a2f, b00f, pacc[2][0]);
                pacc[2][1] = __fmaf_rn(a2f, b01f, pacc[2][1]);
                pacc[2][2] = __fmaf_rn(a2f, b02f, pacc[2][2]);
                pacc[2][3] = __fmaf_rn(a2f, b03f, pacc[2][3]);
                pacc[3][0] = __fmaf_rn(a3f, b00f, pacc[3][0]);
                pacc[3][1] = __fmaf_rn(a3f, b01f, pacc[3][1]);
                pacc[3][2] = __fmaf_rn(a3f, b02f, pacc[3][2]);
                pacc[3][3] = __fmaf_rn(a3f, b03f, pacc[3][3]);
                if (++processed_in_group == GROUP_K || kk_global + 1 == K) flush_group();
            }
        } else {
            #pragma unroll TILE_K
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int kk_global = k0 + kk;
                __half a0 = (row0 + 0 < M) ? sA[stage][(row0 - block_row) + 0][kk] : __float2half(0.f);
                __half a1 = (row0 + 1 < M) ? sA[stage][(row0 - block_row) + 1][kk] : __float2half(0.f);
                __half a2 = (row0 + 2 < M) ? sA[stage][(row0 - block_row) + 2][kk] : __float2half(0.f);
                __half a3 = (row0 + 3 < M) ? sA[stage][(row0 - block_row) + 3][kk] : __float2half(0.f);
                const int bc0 = (col0 - block_col) + 0;
                const int bc1 = bc0 + 1;
                const int bc2 = bc0 + 2;
                const int bc3 = bc0 + 3;
                __half b00 = (col0 + 0 < N) ? sB[stage][kk][bc0] : __float2half(0.f);
                __half b01 = (col0 + 1 < N) ? sB[stage][kk][bc1] : __float2half(0.f);
                __half b02 = (col0 + 2 < N) ? sB[stage][kk][bc2] : __float2half(0.f);
                __half b03 = (col0 + 3 < N) ? sB[stage][kk][bc3] : __float2half(0.f);
                float a0f = __half2float(a0), a1f = __half2float(a1);
                float a2f = __half2float(a2), a3f = __half2float(a3);
                float b00f = __half2float(b00), b01f = __half2float(b01);
                float b02f = __half2float(b02), b03f = __half2float(b03);
                pacc[0][0] = __fmaf_rn(a0f, b00f, pacc[0][0]);
                pacc[0][1] = __fmaf_rn(a0f, b01f, pacc[0][1]);
                pacc[0][2] = __fmaf_rn(a0f, b02f, pacc[0][2]);
                pacc[0][3] = __fmaf_rn(a0f, b03f, pacc[0][3]);
                pacc[1][0] = __fmaf_rn(a1f, b00f, pacc[1][0]);
                pacc[1][1] = __fmaf_rn(a1f, b01f, pacc[1][1]);
                pacc[1][2] = __fmaf_rn(a1f, b02f, pacc[1][2]);
                pacc[1][3] = __fmaf_rn(a1f, b03f, pacc[1][3]);
                pacc[2][0] = __fmaf_rn(a2f, b00f, pacc[2][0]);
                pacc[2][1] = __fmaf_rn(a2f, b01f, pacc[2][1]);
                pacc[2][2] = __fmaf_rn(a2f, b02f, pacc[2][2]);
                pacc[2][3] = __fmaf_rn(a2f, b03f, pacc[2][3]);
                pacc[3][0] = __fmaf_rn(a3f, b00f, pacc[3][0]);
                pacc[3][1] = __fmaf_rn(a3f, b01f, pacc[3][1]);
                pacc[3][2] = __fmaf_rn(a3f, b02f, pacc[3][2]);
                pacc[3][3] = __fmaf_rn(a3f, b03f, pacc[3][3]);
                if (++processed_in_group == GROUP_K || kk_global + 1 == K) flush_group();
            }
        }

        __syncthreads();
        stage ^= 1;
    }

    // Final writeback
    if (row0 + 0 < M) {
        if (col0 + 0 < N) C[(row0 + 0)*N + (col0 + 0)] = gacc[0][0];
        if (col0 + 1 < N) C[(row0 + 0)*N + (col0 + 1)] = gacc[0][1];
        if (col0 + 2 < N) C[(row0 + 0)*N + (col0 + 2)] = gacc[0][2];
        if (col0 + 3 < N) C[(row0 + 0)*N + (col0 + 3)] = gacc[0][3];
    }
    if (row0 + 1 < M) {
        if (col0 + 0 < N) C[(row0 + 1)*N + (col0 + 0)] = gacc[1][0];
        if (col0 + 1 < N) C[(row0 + 1)*N + (col0 + 1)] = gacc[1][1];
        if (col0 + 2 < N) C[(row0 + 1)*N + (col0 + 2)] = gacc[1][2];
        if (col0 + 3 < N) C[(row0 + 1)*N + (col0 + 3)] = gacc[1][3];
    }
    if (row0 + 2 < M) {
        if (col0 + 0 < N) C[(row0 + 2)*N + (col0 + 0)] = gacc[2][0];
        if (col0 + 1 < N) C[(row0 + 2)*N + (col0 + 1)] = gacc[2][1];
        if (col0 + 2 < N) C[(row0 + 2)*N + (col0 + 2)] = gacc[2][2];
        if (col0 + 3 < N) C[(row0 + 2)*N + (col0 + 3)] = gacc[2][3];
    }
    if (row0 + 3 < M) {
        if (col0 + 0 < N) C[(row0 + 3)*N + (col0 + 0)] = gacc[3][0];
        if (col0 + 1 < N) C[(row0 + 3)*N + (col0 + 1)] = gacc[3][1];
        if (col0 + 2 < N) C[(row0 + 3)*N + (col0 + 2)] = gacc[3][2];
        if (col0 + 3 < N) C[(row0 + 3)*N + (col0 + 3)] = gacc[3][3];
    }
}

// Wrapper
void gemm_blockquant_mxint8_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int group_k, int group_n) {
    check_args(A, B, C);
    TORCH_CHECK(group_k == 64 || group_k == 128 || group_k == 256, "group_k must be one of {64,128,256}");
    TORCH_CHECK(group_n == 4 || group_n == 8 || group_n == 16 || group_n == 32,
                "group_n must be one of {4,8,16,32}");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto Cc = C.contiguous();

    const int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    TORCH_CHECK((K % group_k) == 0, "K must be divisible by group_k");
    TORCH_CHECK((N % group_n) == 0, "N must be divisible by group_n");

    dim3 block(BLK_TX, BLK_TY);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    if (group_k == 64) {
        switch (group_n) {
            case 4:  gemm_blockquant_mxint8_core_kernel<64,4><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 8:  gemm_blockquant_mxint8_core_kernel<64,8><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 16: gemm_blockquant_mxint8_core_kernel<64,16><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 32: gemm_blockquant_mxint8_core_kernel<64,32><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
        }
    } else if (group_k == 128) {
        switch (group_n) {
            case 4:  gemm_blockquant_mxint8_core_kernel<128,4><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 8:  gemm_blockquant_mxint8_core_kernel<128,8><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 16: gemm_blockquant_mxint8_core_kernel<128,16><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 32: gemm_blockquant_mxint8_core_kernel<128,32><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
        }
    } else { // 256
        switch (group_n) {
            case 4:  gemm_blockquant_mxint8_core_kernel<256,4><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 8:  gemm_blockquant_mxint8_core_kernel<256,8><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 16: gemm_blockquant_mxint8_core_kernel<256,16><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
            case 32: gemm_blockquant_mxint8_core_kernel<256,32><<<grid, block>>>(
                        reinterpret_cast<const __half*>(Ac.data_ptr<at::Half>()),
                        reinterpret_cast<const __half*>(Bc.data_ptr<at::Half>()),
                        reinterpret_cast<__half*>(Cc.data_ptr<at::Half>()), M, N, K); break;
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_blockquant_mxint8 kernel failed: ", cudaGetErrorString(err));
    if (!C.is_contiguous()) C.copy_(Cc);
}
