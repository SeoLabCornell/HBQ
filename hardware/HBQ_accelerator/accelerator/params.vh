
`ifndef PARAMS_VH
`define PARAMS_VH

// PE
parameter X = 4;  // 3, 4, 5, 6, 7, 8
parameter Y = 5;  // 4, 5, 6, 7, 8
parameter B = 128; // 4, 8, 16, 32, 64, 128
parameter SUB_B = 32;
parameter NUM_SUB_B = B/SUB_B;
parameter PRODUCT_WIDTH = (X-2)+(Y-2)+5;
parameter LEVEL = $clog2(B);
parameter SUB_BLOCK_LEVEL = $clog2(SUB_B);
parameter L2_SCALE_X_SCHEME = 0; // 0 for PoT, 1 for INT
parameter L2_SCALE_X_BIT = 1;
parameter L2_SCALE_W_BIT = 2;
parameter L2_SCALE_X_SHIFT = (L2_SCALE_X_SCHEME == 1) ? L2_SCALE_X_BIT : (1 << L2_SCALE_X_BIT)-1; // if integer then = L2_SCALE_X_BIT, if PoT then = (2**L2_SCALE_X_BIT)-1
parameter L2_SCALE_W_SHIFT = 2+L2_SCALE_W_BIT; // using FP-1 and FP-10, so at most add 2-bit for 1.0XX
parameter L2_PSUM_WIDTH = PRODUCT_WIDTH + SUB_BLOCK_LEVEL;
parameter L2_ADDER_TREE_LEVEL = $clog2(NUM_SUB_B);

parameter SCALED_L2_PSUM_WIDTH = L2_PSUM_WIDTH + L2_SCALE_W_SHIFT + L2_SCALE_X_SHIFT; // 4 bit for wgt 2b-1.0XX, 2 bit for act 2b-INT
parameter FX_PSUM_WIDTH = SCALED_L2_PSUM_WIDTH+L2_ADDER_TREE_LEVEL; // fixed point psum

parameter MODE_128 = 1;
parameter MODE_L2 = 2;


// parameter L2_SCHEME_INT = 0;
parameter L2_SCHEME_FP = 0;
parameter L2_SCHEME_FP_10 = 1;

// SA
parameter SA_N = 16; // activation propogate along this dimension

// Psum quantizer
parameter PSUM_QUANTIZATION_B = SA_N;
parameter PSUM_WIDTH = 16; // FP16
parameter PSUM_EXP_WIDTH = 5; // FP16
parameter PSUM_MANT_WIDTH = 10; // FP16
parameter PSUM_EXP_BIAS = 15; // FP16
parameter MX_SCALE_WIDTH = 8; // MXINT8
parameter PSUM_MX_WIDTH = 8; // MXINT8
parameter MX_LP_EXP_MAX = MX_SCALE_WIDTH-2; // MXINT8

// Act buffer
parameter TILE_TOKEN = 128;
parameter ACT_BUFFER_WORD_WIDTH = 640;
parameter ACT_BUFFER_DEPTH = TILE_TOKEN;
parameter ACT_BUFFER_ADDR_WIDTH = $clog2(ACT_BUFFER_DEPTH);

parameter ACT_RF_BANKS = 4;
parameter ACT_RF_WORD_WIDTH = 160;
parameter ACT_RF_DEPTH = TILE_TOKEN;
parameter ACT_RF_ADDR_WIDTH = $clog2(ACT_RF_DEPTH);

parameter ACT_SCALE_RF_WORD_WIDTH = 24;
parameter ACT_SCALE_RF_DEPTH = TILE_TOKEN;
parameter ACT_SCALE_RF_ADDR_WIDTH = $clog2(ACT_SCALE_RF_DEPTH);

// Psum buffer
parameter SA_PP_STAGE = 2;
parameter TILE_PSUM = 256;
parameter PSUM_BUFFER_WORD_WIDTH = PSUM_MX_WIDTH*SA_N+8; // 136-bit, 8bit for scaling factor
parameter PSUM_BUFFER_DEPTH = TILE_PSUM/SA_N*TILE_TOKEN; // 256/16*128 = 2048
parameter PSUM_BUFFER_ADDR_WIDTH = $clog2(PSUM_BUFFER_DEPTH); // log2(2048) = 11

parameter PSUM_RF_BANKS = 8;
parameter PSUM_RF_WORD_WIDTH = PSUM_BUFFER_WORD_WIDTH;
parameter PSUM_RF_DEPTH = PSUM_BUFFER_DEPTH/PSUM_RF_BANKS; // 2048/8 = 256
parameter PSUM_RF_ADDR_WIDTH = $clog2(PSUM_RF_DEPTH); // log2(256) = 8
parameter PSUM_RF_BANK_ADDR_WIDTH = $clog2(PSUM_RF_BANKS); // log2(8) = 3

`endif