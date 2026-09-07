
`ifndef PARAMS_VH
`define PARAMS_VH

parameter integer X = 4;  // 3, 4, 5, 6, 7, 8
parameter integer Y = 5;  // 4, 5, 6, 7, 8
parameter integer B = 128; // 4, 8, 16, 32, 64, 128
parameter integer SUB_B = 8;
parameter integer NUM_SUB_B = B/SUB_B;
parameter PRODUCT_WIDTH = (X-2)+(Y-2)+5;
parameter LEVEL = $clog2(B);
parameter SUB_BLOCK_LEVEL = $clog2(SUB_B);
parameter L2_SCALE_X_SCHEME = 2; // 0 for PoT, 1 for INT, 2 for SIG-1
parameter L2_SCALE_W_SCHEME = 1; // 0 for INT, 1 for SIG
parameter L2_SCALE_X_BIT = 2;
parameter L2_SCALE_W_BIT = 2;
parameter L2_SCALE_X_SHIFT = (L2_SCALE_X_SCHEME == 0) ? (1 << L2_SCALE_X_BIT)-1 : (L2_SCALE_X_SCHEME == 1) ? L2_SCALE_X_BIT : 2; // if integer then = L2_SCALE_X_BIT, if PoT then = (2**L2_SCALE_X_BIT)-1, SIG-1 leading 1 and sign bit
// parameter L2_SCALE_W_SHIFT = (L2_SCALE_W_SCHEME == 1) ? 2+L2_SCALE_W_BIT : L2_SCALE_W_BIT; // using FP-1 and FP-10, so at most add 2-bit for 1.0XX
parameter L2_SCALE_W_SHIFT = 2; // leading 1 + sign bit
parameter L2_PSUM_WIDTH = PRODUCT_WIDTH + SUB_BLOCK_LEVEL;
parameter L2_ADDER_TREE_LEVEL = $clog2(NUM_SUB_B);

parameter SCALED_L2_PSUM_WIDTH = L2_PSUM_WIDTH + L2_SCALE_W_SHIFT + L2_SCALE_X_SHIFT; // 4 bit for wgt 2b-1.0XX, 2 bit for act 2b-INT, add 1-bit sign
parameter PSUM_WIDTH = SCALED_L2_PSUM_WIDTH+L2_ADDER_TREE_LEVEL;

parameter MODE_64 = 1;
parameter MODE_128 = 2;
parameter MODE_L2  = 3;


// parameter L2_SCHEME_INT = 0;
parameter L2_SCHEME_FP = 0;
parameter L2_SCHEME_FP_10 = 1;


`endif