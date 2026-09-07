`timescale 1ns / 1ps
`include "get_scale.v"
//////////////////////////////////////////////////////////////////////////////////
// Floating-Point to NVFP Converter                                             //
// ---------------------------------------------------------------------------  //
//  * Parameter-free top file: choose options with compile-time `define`s below //
//////////////////////////////////////////////////////////////////////////////////
//============================ USER-CONFIGURABLE MACROS ==========================
// >>> Select group size (supported: 16 or 32) <<<
`define BLOCK_SIZE 128
`define MICRO_BLOCK_SIZE 32
//==============================================================================
//--------------------------- Fixed-width parameters ----------------------------
// Using FP16 -> (S0E5M3 scale, S1E2M1 element) NVFP4 conversion
// For other precision, change parameters below accordingly 
`define FP_DATAWIDTH 16
`define FP_SIGN      1
`define FP_EXP       5
`define FP_MANT      10
`define FP_EXP_BIAS  15
`define LP_DATAWIDTH 5
`define LP_EXP_MAX   2
`define LP_EXP_BIAS  1
`define SCALE_DATAWIDTH 8
`define SCALE_EXP_BIAS  15
`define SCALE_EXP       5
`define SCALE_MANT      3
// LUT mantissa precision for 1/scale
`define LUT_WIDTH 4
//////////////////////////////////////////////////////////////////////////////////
//  MODULE : FP2GS  (FP-to-Grouped-Scaled converter)                            //
//////////////////////////////////////////////////////////////////////////////////
module FP2GS #(
    parameter GROUP_N = `BLOCK_SIZE,
    parameter NUM_MICRO_BLOCKS = `BLOCK_SIZE / `MICRO_BLOCK_SIZE,
    parameter MANT_BIT_POST_L1 = 5
)(
    input clk,
    input rst_n,
    input  [`FP_DATAWIDTH*GROUP_N-1:0] FP_DATA,
    output [`LP_DATAWIDTH*GROUP_N-1:0] LP_DATA,
    output [`SCALE_DATAWIDTH-1:0]      L1_SCALE,
    output [NUM_MICRO_BLOCKS-1:0]      L2_SCALE
);
// -----------------------------------------------------------------------------
// 1) Flattened FP_DATA → per-element vectors, find scale
// -----------------------------------------------------------------------------
// wire [`FP_DATAWIDTH-1:0]          fp_vec      [0:GROUP_N-1];
reg  [`FP_SIGN-1:0]               fp_sign_r     [0:GROUP_N-1];
reg  [`FP_EXP-1:0]                fp_exp_r      [0:GROUP_N-1];
reg  [`FP_MANT-1:0]               fp_mant_r     [0:GROUP_N-1];
reg  [`FP_SIGN-1:0]               fp_sign_w     [0:GROUP_N-1];
reg  [`FP_EXP-1:0]                fp_exp_w      [0:GROUP_N-1];
reg  [`FP_MANT-1:0]               fp_mant_w     [0:GROUP_N-1];
reg  [`SCALE_DATAWIDTH-1:0]       l1_scale_1_r, l1_scale_1_w;
// Split packed input into parallel fields, pipeline one cycle for scale calculation
integer gi;
always @(*) begin
    for (gi = 0; gi < GROUP_N; gi = gi + 1) begin : SPLIT
        {fp_sign_w[gi], fp_exp_w[gi], fp_mant_w[gi]} = FP_DATA[`FP_DATAWIDTH*(gi+1)-1-:`FP_DATAWIDTH];
    end
end

GET_L1_SCALE #(
    .FP_DATAWIDTH(`FP_DATAWIDTH),
    .FP_SIGN(`FP_SIGN),
    .FP_EXP(`FP_EXP),
    .FP_MANT(`FP_MANT),
    .SCALE_EXP(`SCALE_EXP),
    .SCALE_MANT(`SCALE_MANT),
    .SCALE_DATAWIDTH(`SCALE_DATAWIDTH),
    .GROUP_N(GROUP_N)
) get_l1_scale_inst (
    .FP_DATA(FP_DATA),
    .LP_SCALE(l1_scale_1_w)
);

always @(posedge clk or negedge rst_n) begin // one stage pipeline for scale calculation
    if (!rst_n) begin
        for (gi = 0; gi < GROUP_N; gi = gi + 1) begin
            fp_sign_r[gi] <= 0;
            fp_exp_r[gi] <= 0;
            fp_mant_r[gi] <= 0;
        end
        l1_scale_1_r <= 0;
    end else begin
        for (gi = 0; gi < GROUP_N; gi = gi + 1) begin
            fp_sign_r[gi] <= fp_sign_w[gi];
            fp_exp_r[gi] <= fp_exp_w[gi];
            fp_mant_r[gi] <= fp_mant_w[gi];
        end
        l1_scale_1_r <= l1_scale_1_w;
    end
end
// -----------------------------------------------------------------------------
// 2) x/scale, use LUT for 1/scale
//    (placeholder below keeps synthesis happy)
// -----------------------------------------------------------------------------
// use 5 bit for reciprocal LUT

wire [`SCALE_EXP-1:0] scale_exp;
wire [`SCALE_MANT-1:0] scale_mant;
assign scale_exp = l1_scale_1_r[`SCALE_DATAWIDTH-1:`SCALE_DATAWIDTH-`SCALE_EXP];
assign scale_mant = l1_scale_1_r[`SCALE_DATAWIDTH-`SCALE_EXP-1:0];
reg [`LUT_WIDTH-1:0] scale_lut [0:7]; // 8 entries for M3, 2/1.M3
always @(posedge clk) begin
    scale_lut[0] = 4'b0000;
    scale_lut[1] = 4'b1100;
    scale_lut[2] = 4'b1010;
    scale_lut[3] = 4'b0111;
    scale_lut[4] = 4'b0101;
    scale_lut[5] = 4'b0100;
    scale_lut[6] = 4'b0010;
    scale_lut[7] = 4'b0001;
end

wire [`LUT_WIDTH-1:0] scale_inv_mant;
assign scale_inv_mant = scale_lut[scale_mant];


reg [`FP_MANT+`LUT_WIDTH+1:0] lp_mant_product [0:GROUP_N-1]; // add 2 for leading 1
reg signed [`FP_EXP-1:0] fp_exp_post_l1_scale [0:GROUP_N-1]; // unbiased exp
reg signed [`FP_EXP-1:0] scale_diff [0:GROUP_N-1]; // unbiased exp

reg [`FP_MANT:0] lp_mant_round [0:GROUP_N-1];

integer idx;
always @(*) begin
    for (idx = 0; idx < GROUP_N; idx = idx + 1) begin : sLP
        scale_diff[idx] = fp_exp_r[idx] - scale_exp;
        if (scale_mant == 0) begin // 2/1.0 -> give exponent one more
            fp_exp_post_l1_scale[idx] = scale_diff[idx]; // scale bias is the same as fp bias, canceled out here
        end else begin // 2/1.x -> mant guaranteed to be in range of (2,1)
            fp_exp_post_l1_scale[idx] = scale_diff[idx] - 1; // scale bias is the same as fp bias, canceled out here
        end
        // mant multiplication with LUT (*2/1.M3)
        lp_mant_product[idx] = {1'b1, fp_mant_r[idx]} * {1'b1, scale_inv_mant}; // product can be 11.xxxx or 10.xxxx or 01.xxxx
        if (lp_mant_product[idx][`FP_MANT+`LUT_WIDTH+1]) begin // 11.xxxx or 10.xxxx
            fp_exp_post_l1_scale[idx] = fp_exp_post_l1_scale[idx] + 1;
            lp_mant_round[idx] = lp_mant_product[idx][`FP_MANT+`LUT_WIDTH+1-1-:`FP_MANT];
        end else begin // 01.xxxx
            lp_mant_round[idx] = lp_mant_product[idx][`FP_MANT+`LUT_WIDTH+1-2-:`FP_MANT];
        end
    end
end


// -----------------------------------------------------------------------------
// 3) Second level scale: 
// -----------------------------------------------------------------------------
wire [NUM_MICRO_BLOCKS-1:0] l2_scale;
wire [`FP_DATAWIDTH*`MICRO_BLOCK_SIZE-1:0] fp_data_for_l2 [0:NUM_MICRO_BLOCKS-1];
genvar micro_block_idx;
genvar element_idx;
generate
    for (micro_block_idx = 0; micro_block_idx < NUM_MICRO_BLOCKS; micro_block_idx = micro_block_idx + 1) begin : FP_DATA_FOR_L2
        for (element_idx = 0; element_idx < `MICRO_BLOCK_SIZE; element_idx = element_idx + 1) begin : ELEMENT_DATA
            assign fp_data_for_l2[micro_block_idx][element_idx*`FP_DATAWIDTH+:`FP_DATAWIDTH] = {fp_exp_post_l1_scale[micro_block_idx*`MICRO_BLOCK_SIZE+element_idx], lp_mant_round[micro_block_idx*`MICRO_BLOCK_SIZE+element_idx][`FP_MANT:1]};
        end
        GET_L2_SCALE #(
            .FP_DATAWIDTH(`FP_DATAWIDTH),
            .FP_SIGN(`FP_SIGN),
            .FP_EXP(`FP_EXP),
            .FP_MANT(`FP_MANT),
            .FP_EXP_BIAS(`FP_EXP_BIAS),
            .GROUP_N(`MICRO_BLOCK_SIZE)
        ) get_l2_scale_inst (
            .FP_DATA(fp_data_for_l2[micro_block_idx]),
            .LP_SCALE(l2_scale[micro_block_idx])
        );
    end
endgenerate
reg  [`SCALE_DATAWIDTH-1:0] l1_scale_2_r;
reg  [NUM_MICRO_BLOCKS-1:0] l2_scale_r;
reg  [`FP_SIGN-1:0]         fp_sign_post_l1_scale_r       [0:GROUP_N-1];
reg signed [`FP_EXP-1:0]    fp_exp_post_l1_scale_r        [0:GROUP_N-1];
reg [MANT_BIT_POST_L1-1:0]  lp_mant_round_post_l1_scale_r [0:GROUP_N-1]; // keep only 5 bit

always @(posedge clk or negedge rst_n) begin // one stage pipeline for scale calculation
    if (!rst_n) begin
        for (gi = 0; gi < GROUP_N; gi = gi + 1) begin
            fp_sign_post_l1_scale_r[gi] <= 0;
            fp_exp_post_l1_scale_r[gi] <= 0;
            lp_mant_round_post_l1_scale_r[gi] <= 0;
        end
        l1_scale_2_r <= 0;
        l2_scale_r <= 0;
    end else begin
        for (gi = 0; gi < GROUP_N; gi = gi + 1) begin
            fp_sign_post_l1_scale_r[gi] <= fp_sign_r[gi];
            fp_exp_post_l1_scale_r[gi] <= fp_exp_post_l1_scale[gi];
            lp_mant_round_post_l1_scale_r[gi] <= lp_mant_round[gi][`FP_MANT-:MANT_BIT_POST_L1];
        end
        l1_scale_2_r <= l1_scale_1_r;
        l2_scale_r <= l2_scale;
    end
end

// -----------------------------------------------------------------------------
// 4) /L2 scale, quantize to FP5
// -----------------------------------------------------------------------------
reg [`LP_DATAWIDTH-1:0] lp_s1e2m2 [0:`BLOCK_SIZE-1];
reg signed [`FP_EXP-1:0] biased_lp_exp [0:`BLOCK_SIZE-1]; // unbiased exp
reg signed [`FP_EXP:0] fp_exp_post_l2_scale [0:NUM_MICRO_BLOCKS-1][0:`MICRO_BLOCK_SIZE-1]; // unbiased exp
reg is_zero [0:`BLOCK_SIZE-1];
integer ele_idx;
always @(*) begin
    integer idx_1d;
    for (idx = 0; idx < NUM_MICRO_BLOCKS; idx = idx + 1) begin : sLP
        for (ele_idx = 0; ele_idx < `MICRO_BLOCK_SIZE; ele_idx = ele_idx + 1) begin : ELEMENT_DATA
            idx_1d = idx*`MICRO_BLOCK_SIZE+ele_idx;
            is_zero[idx_1d] = (fp_exp_post_l1_scale_r[idx_1d] == 0);
            fp_exp_post_l2_scale[idx][ele_idx] = fp_exp_post_l1_scale_r[idx_1d]-l2_scale_r[idx];

            if (is_zero[idx_1d]) begin
                lp_s1e2m2[idx_1d] = 5'b00000;
            end else if (fp_exp_post_l2_scale[idx][ele_idx] <= -4) begin // clip to 0
                lp_s1e2m2[idx_1d] = 5'b00000;
            end else if (fp_exp_post_l2_scale[idx][ele_idx] == -3) begin // round up to subnormal: 1.xx*2^-3 = 0.001xx*2^0 round up to 0.01
                lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], 4'b0001};
            end else if (fp_exp_post_l2_scale[idx][ele_idx] == -2) begin // subnormal
                if (lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1]) begin // 1.1*2^-2
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], 4'b0010};
                end
                else begin
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], 4'b0001};
                end
            end else if (fp_exp_post_l2_scale[idx][ele_idx] == -1) begin // subnormal
                if (lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1]) begin // 1.1*2^-1 -> promote to normal 1.0*2^0
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], 4'b0100};
                end else begin // subnormal, 1.0x*2^-1
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], 4'b0001};
                end
            end else begin // normal
                if (lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1-:3] == 3'b111) begin // 1.111x case, exp+1
                    if (fp_exp_post_l2_scale[idx][ele_idx] == `LP_EXP_MAX) begin // clip to FP5 max
                        lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], 4'b111};
                    end else begin
                        biased_lp_exp[idx_1d] = fp_exp_post_l2_scale[idx][ele_idx]+1+`LP_EXP_BIAS;
                        lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], biased_lp_exp[idx_1d][1:0], 1'b0};
                    end
                end else if (lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1-:3] == 3'b110 || lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1-:3] == 3'b011) begin // 1.110x or 1.011x
                    biased_lp_exp[idx_1d] = fp_exp_post_l2_scale[idx][ele_idx]+`LP_EXP_BIAS;
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], biased_lp_exp[idx_1d][1:0], 2'b11};
                end else if (lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1-:3] == 3'b101 || lp_mant_round_post_l1_scale_r[idx_1d][MANT_BIT_POST_L1-1-:3] == 3'b010) begin // 1.010x or 1.001x
                    biased_lp_exp[idx_1d] = fp_exp_post_l2_scale[idx][ele_idx]+`LP_EXP_BIAS;
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], biased_lp_exp[idx_1d][1:0], 2'b01};
                end else begin // 1.000x
                    biased_lp_exp[idx_1d] = fp_exp_post_l2_scale[idx][ele_idx]+`LP_EXP_BIAS;
                    lp_s1e2m2[idx_1d] = {fp_sign_post_l1_scale_r[idx_1d], biased_lp_exp[idx_1d][1:0], 2'b00};
                end
            end
        end
    end
end



// output
assign L1_SCALE = l1_scale_2_r;
assign L2_SCALE = l2_scale_r;

genvar lp_idx;
generate
    for (lp_idx = 0; lp_idx < GROUP_N; lp_idx = lp_idx + 1) begin : LP_DATA_GEN
        assign LP_DATA[`LP_DATAWIDTH*lp_idx+:`LP_DATAWIDTH] = lp_s1e2m2[lp_idx];
    end
endgenerate
endmodule