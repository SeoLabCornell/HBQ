`timescale 1ns / 1ps
`include "max_finder.v"
//////////////////////////////////////////////////////////////////////////////////
//  MODULE : FP2GS  (FP-to-Grouped-Scaled converter)                            //
//////////////////////////////////////////////////////////////////////////////////
module GET_GROUP_SCALE #(
    parameter FP_DATAWIDTH = 16,
    parameter FP_SIGN = 1,
    parameter FP_EXP = 5,
    parameter FP_MANT = 10,
    parameter SCALE_EXP = 5,
    parameter SCALE_MANT = 10,
    parameter SCALE_DATAWIDTH = 8,
    parameter LP_EXP_MAX = 2,
    parameter GROUP_N = 16
)(
    input  USE_NVFP,
    input  [FP_DATAWIDTH*GROUP_N-1:0] FP_DATA,
    output [SCALE_DATAWIDTH-1:0]      LP_SCALE
);
// -----------------------------------------------------------------------------
// 1) Flattened FP_DATA → per-element vectors
// -----------------------------------------------------------------------------
wire [FP_DATAWIDTH-1:0]          fp_vec      [0:GROUP_N-1];
wire [FP_SIGN-1:0]               fp_sign     [0:GROUP_N-1];
wire [FP_EXP-1:0]                fp_exp      [0:GROUP_N-1];
wire [FP_MANT-1:0]               fp_mant     [0:GROUP_N-1];
wire [FP_MANT-1:0]               fp_mant_normal     [0:GROUP_N-1];
wire [FP_DATAWIDTH-2:0]          fp_cmp      [0:GROUP_N-1];
// Split packed input into parallel fields
genvar gi;
generate
    for (gi = 0; gi < GROUP_N; gi = gi + 1) begin : SPLIT
        assign fp_vec[gi]  = FP_DATA[FP_DATAWIDTH*(gi+1)-1:FP_DATAWIDTH*gi];
        assign {fp_sign[gi], fp_exp[gi], fp_mant[gi]} = fp_vec[gi];
//        assign fp_mant_normal[gi] = (fp_exp[gi]==0) ? 0 : {1'b1, fp_mant[gi]}; // ignore subnormal case
        // Remove sign bit, take the rest for comparison
        assign fp_cmp[gi] = fp_vec[gi][FP_DATAWIDTH-2:0];
    end
endgenerate
// -----------------------------------------------------------------------------
// 2) Shared-scale search (max of fp_cmp[])
// -----------------------------------------------------------------------------
localparam SEG = GROUP_N / 8; // each MaxExpFinder consumes 8 inputs
wire [FP_DATAWIDTH-2:0] local_max [0:SEG-1];
// Level-1 reduction: max of every 8 elements
generate
    for (gi = 0; gi < SEG; gi = gi + 1) begin : LOCAL_MAX_GEN
        max_finder #(.BIT_WIDTH(FP_DATAWIDTH-1)) MF (
            .data({ fp_cmp[gi*8+0], fp_cmp[gi*8+1], fp_cmp[gi*8+2], fp_cmp[gi*8+3],
                     fp_cmp[gi*8+4], fp_cmp[gi*8+5], fp_cmp[gi*8+6], fp_cmp[gi*8+7] }),
            .max (local_max[gi])
        );
    end
endgenerate
// Level-2 reduction: max over SEG partial maxima
reg [FP_DATAWIDTH-2:0] global_max_r;
integer ii;
always @* begin
    global_max_r = local_max[0];
    for (ii = 1; ii < SEG; ii = ii + 1) begin
        if (local_max[ii] > global_max_r)
            global_max_r = local_max[ii];
    end
end
// -----------------------------------------------------------------------------
// 3) scale 
// NV scale = max element * 1/6
// MX scale = exp_max (will ele_exp_max in top.sv)
// -----------------------------------------------------------------------------
localparam [SCALE_DATAWIDTH-1:0] NVFP_MIN_SCALE = 8'b00001000;
wire [FP_EXP-1:0] max_exp;
wire [FP_MANT-1:0] max_mant;
wire [21:0] mant_product; // use 8bit for max_mant, 8bit for elemax_mant, plus the leading 1
wire [3:0]  mant_product_round; // for mantissa rounding, m3+1bit for rounding
wire        round_up; // for mantissa rounding
reg  signed [SCALE_EXP:0] scale_exp; // add 1 bit for sign to avoid underflow
reg  [SCALE_MANT-1:0] scale_mant;

assign max_exp = global_max_r[FP_DATAWIDTH-2-:FP_EXP];
assign max_mant = global_max_r[FP_DATAWIDTH-FP_EXP-2-:FP_MANT];

// FP4 max = 6, 1/6 = 1.01010101 * 2e-3, use 10bit mantissa for now
localparam [9:0] elemax_mant = 10'b0101010101;
localparam [1:0] elemax_exp_shift = 2'd3;

assign mant_product = {1'b1, max_mant[FP_MANT-1-:10]} * {1'b1, elemax_mant}; // 1.xxxx * 1.0101 = 01.xxxx or 10.xxxx
assign scale_exp = {mant_product[21]} ? $signed({1'b0, max_exp}) - $signed({1'b0, elemax_exp_shift})+1 : $signed({1'b0, max_exp}) - $signed({1'b0, elemax_exp_shift}); // if max_mant is 1.1x, 1/6 will be larger than 1/4, so can shift 1 bit less
assign mant_product_round = {mant_product[21]} ? mant_product[20:17] : mant_product[19:16]; // if max_mant is 1.1x, product will be 10.x, else product will be 01.x
assign scale_mant = {mant_product_round[0]} ? mant_product_round[3:1]+1 : mant_product_round[3:1]; 
assign round_up = mant_product_round == 4'b1111;

assign LP_SCALE = USE_NVFP ? 
        (($signed(scale_exp+round_up) < $signed(1)) ? NVFP_MIN_SCALE : {scale_exp[SCALE_EXP-1:0]+round_up, scale_mant}) : // don't need to add bias here since FP16 shares same bias with E5M3
        max_exp;

endmodule