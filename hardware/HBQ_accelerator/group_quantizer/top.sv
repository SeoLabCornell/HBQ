`timescale 1ns / 1ps
`include "get_group_scale.v"
//////////////////////////////////////////////////////////////////////////////////
// Floating-Point to Group low FP Converter                                     //
// Now supports NVFP4 and MXFP4 for group quantization conversion               //
// ---------------------------------------------------------------------------  //
//  * Parameter-free top file: choose options with compile-time `define`s below //
//  * Supports GROUP_NUMBER 16 or 32                                            //
//  * Use input USE_NVFP to select NVFP or MXFP conversion                      //
//////////////////////////////////////////////////////////////////////////////////
//============================ USER-CONFIGURABLE MACROS ==========================
// >>> Select group size (supported: 16 or 32) <<<
`define GROUP_NUMBER 16
//==============================================================================
//--------------------------- Fixed-width parameters ----------------------------
// Using FP16 -> (S0E5M3 scale, S1E2M1 element) NVFP4 conversion or (S0E5M0 scale, S1E2M1 element) MXFP4 conversion
// For other precision, change parameters below accordingly 
`define FP_DATAWIDTH 16
`define FP_SIGN      1
`define FP_EXP       5
`define FP_MANT      10
`define FP_EXP_BIAS  15
`define LP_DATAWIDTH 4
`define LP_EXP_MAX   2
`define LP_EXP_BIAS  1
`define SCALE_DATAWIDTH 8
`define SCALE_EXP_BIAS  15
`define SCALE_EXP       5
`define SCALE_MANT      3
// LUT mantissa precision for 1/scale
`define LUT_WIDTH 4
// USE_NVFP: NVFP or MXFP
//////////////////////////////////////////////////////////////////////////////////
//  MODULE : FP2GS  (FP-to-Grouped-Scaled converter)                            //
//////////////////////////////////////////////////////////////////////////////////
module FP2GS #(
    parameter GROUP_N = `GROUP_NUMBER
)(
    input  clk,
    input  USE_NVFP, // 1: NVFP, 0: MXFP
    input  [`FP_DATAWIDTH*GROUP_N-1:0] FP_DATA,
    output [`LP_DATAWIDTH*GROUP_N-1:0] LP_DATA,
    output [`SCALE_DATAWIDTH-1:0]      LP_SCALE
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
reg  [`SCALE_DATAWIDTH-1:0]       scale_r;
reg  [`SCALE_DATAWIDTH-1:0]       scale_w;
// Split packed input into parallel fields, pipeline one cycle for scale calculation
integer gi;
always @(*) begin
    for (gi = 0; gi < GROUP_N; gi = gi + 1) begin : SPLIT
        {fp_sign_w[gi], fp_exp_w[gi], fp_mant_w[gi]} = FP_DATA[`FP_DATAWIDTH*(gi+1)-1-:`FP_DATAWIDTH];
    end
end

GET_GROUP_SCALE #(
    .FP_DATAWIDTH(`FP_DATAWIDTH),
    .FP_SIGN(`FP_SIGN),
    .FP_EXP(`FP_EXP),
    .FP_MANT(`FP_MANT),
    .SCALE_EXP(`SCALE_EXP),
    .SCALE_MANT(`SCALE_MANT),
    .SCALE_DATAWIDTH(`SCALE_DATAWIDTH),
    .LP_EXP_MAX(`LP_EXP_MAX),
    .GROUP_N(GROUP_N)
) get_nv_scale_inst (
    .USE_NVFP(USE_NVFP),
    .FP_DATA(FP_DATA),
    .LP_SCALE(scale_w)
);

always @(posedge clk) begin // one stage pipeline for scale calculation
    for (gi = 0; gi < GROUP_N; gi = gi + 1) begin
        fp_sign_r[gi] <= fp_sign_w[gi];
        fp_exp_r[gi] <= fp_exp_w[gi];
        fp_mant_r[gi] <= fp_mant_w[gi];
    end
    scale_r <= scale_w;
end
// -----------------------------------------------------------------------------
// 2-1) NVFP
//    x/scale, use LUT for 1/scale
//    (placeholder below keeps synthesis happy)
// -----------------------------------------------------------------------------
// use 5 bit for reciprocal LUT
wire [`SCALE_EXP-1:0] scale_exp;
wire [`SCALE_MANT-1:0] scale_mant;
assign scale_exp = scale_r[`SCALE_DATAWIDTH-1:`SCALE_DATAWIDTH-`SCALE_EXP];
assign scale_mant = scale_r[`SCALE_DATAWIDTH-`SCALE_EXP-1:0];
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

reg is_zero [0:GROUP_N-1];
reg [`FP_MANT+`LUT_WIDTH+1:0] lp_mant_product [0:GROUP_N-1]; // add 2 for leading 1
reg signed [`FP_EXP:0] lp_exp [0:GROUP_N-1]; // unbiased exp, add one bit for underflow
reg signed [`FP_EXP:0] scale_diff [0:GROUP_N-1]; // unbiased exp
reg signed [`FP_EXP:0] biased_lp_exp [0:GROUP_N-1]; // unbiased exp
reg [1:0] lp_mant_round [0:GROUP_N-1]; // M1 + 1 bit for rounding, e.g. 1.lp_mant_round(2b)
reg [`LP_DATAWIDTH-1:0] nvfp_s1e2m1 [0:GROUP_N-1];
integer idx;
always @(*) begin
    for (idx = 0; idx < GROUP_N; idx = idx + 1) begin : sLP
        is_zero[idx] = (fp_exp_r[idx] == 0);
        scale_diff[idx] = fp_exp_r[idx] - scale_exp;
        if (scale_mant == 0) begin // 2/1.0 -> give exponent one more
            lp_exp[idx] = scale_diff[idx]; // scale bias is the same as fp bias, canceled out here
        end else begin // 2/1.x -> mant guaranteed to be in range of (2,1)
            lp_exp[idx] = scale_diff[idx] - 1; // scale bias is the same as fp bias, canceled out here
        end
        // mant multiplication
        lp_mant_product[idx] = {1'b1, fp_mant_r[idx]} * {1'b1, scale_inv_mant}; // product can be 11.xxxx or 10.xxxx or 01.xxxx
        if (lp_mant_product[idx][`FP_MANT+`LUT_WIDTH+1]) begin // 11.xxxx or 10.xxxx
            lp_exp[idx] = lp_exp[idx] + 1;
            lp_mant_round[idx] = lp_mant_product[idx][`FP_MANT+`LUT_WIDTH+1-1-:2];
        end else begin // 01.xxxx
            lp_mant_round[idx] = lp_mant_product[idx][`FP_MANT+`LUT_WIDTH+1-2-:2];
        end

        if (is_zero[idx]) begin
            nvfp_s1e2m1[idx] = 4'b0000;
        end else if (lp_exp[idx] <= -3) begin // clip to 0
            nvfp_s1e2m1[idx] = 4'b0000;
        end else if (lp_exp[idx] == -2) begin // round up to subnormal: 1.xx*2^-2 = 0.01xx*2^0 round up to 0.1*2^-1
            nvfp_s1e2m1[idx] = {fp_sign_r[idx], 3'b001};
        end else if (lp_exp[idx] == -1) begin // subnormal
            if (lp_mant_round[idx][1]) begin // 1.1x*2^-1 -> promote to normal 1.0*2^0
                nvfp_s1e2m1[idx] = {fp_sign_r[idx], 3'b010};
            end else begin // subnormal, 1.0x*2^-1
                nvfp_s1e2m1[idx] = {fp_sign_r[idx], 3'b001};
            end
        end else begin // normal
            if (lp_mant_round[idx] == 2'b11) begin // 1.11x case, exp+1
                if (lp_exp[idx] == `LP_EXP_MAX) begin // clip to FP4 max (6)
                    nvfp_s1e2m1[idx] = {fp_sign_r[idx], 3'b111};
                end else begin
                    biased_lp_exp[idx] = lp_exp[idx]+1+`LP_EXP_BIAS;
                    nvfp_s1e2m1[idx] = {fp_sign_r[idx], biased_lp_exp[idx][1:0], 1'b0};
                end
            end else if (lp_mant_round[idx][1]|lp_mant_round[idx][0]) begin // 1.10x or 1.01x
                biased_lp_exp[idx] = lp_exp[idx]+`LP_EXP_BIAS;
                nvfp_s1e2m1[idx] = {fp_sign_r[idx], biased_lp_exp[idx][1:0], 1'b1};
            end else begin // 1.00x
                biased_lp_exp[idx] = lp_exp[idx]+`LP_EXP_BIAS;
                nvfp_s1e2m1[idx] = {fp_sign_r[idx], biased_lp_exp[idx][1:0], 1'b0};
            end
        end
    end
end

// -----------------------------------------------------------------------------
// 2-2) MXFP (FP16 -> FP4) shift and round
//    Quantise mantissa to L-precision using LP_SCALE
// -----------------------------------------------------------------------------
genvar iidx;
wire [`FP_EXP-1:0] shift_LP [0:GROUP_N-1];
wire [`FP_MANT+1+`LP_EXP_MAX-1:0] LP_MANT_H_shifted [0:GROUP_N-1]; // +1 for leading 1, +`LP_EXP_MAX for exp_max
wire [4:0] LP_MANT_H [0:GROUP_N-1]; /// E2M1 pre-norm
wire [2:0] LP_E2M1 [0:GROUP_N-1];
wire [3:0] mxfp_s1e2m1 [0:GROUP_N-1];
generate
    for (iidx = 0; iidx < GROUP_N; iidx = iidx + 1) begin : MXFP_LP_GEN
        assign shift_LP[iidx] = (scale_r <= `LP_EXP_MAX) ? 1+`LP_EXP_MAX-fp_exp_r[iidx] : scale_r-fp_exp_r[iidx];
        assign LP_MANT_H_shifted[iidx] = (fp_exp_r[iidx] == 0) ? {1'b0, fp_mant_r[iidx], {`LP_EXP_MAX{1'b0}}} >> shift_LP[iidx] : {1'b1, fp_mant_r[iidx], {`LP_EXP_MAX{1'b0}}} >> shift_LP[iidx]; // ignore fp16 subnormal
        assign LP_MANT_H[iidx] = LP_MANT_H_shifted[iidx][`FP_MANT+1+`LP_EXP_MAX-1-:5]; /// case for E2M1. it can cover 4bit width + 1 bit for rounding, {leading 1, }
        assign LP_E2M1[iidx] =   LP_MANT_H[iidx][4] ? {2'b11, LP_MANT_H[iidx][3]||LP_MANT_H[iidx][2]} :  ///// 5'b1.1xxx
                                LP_MANT_H[iidx][3] ? ( LP_MANT_H[iidx][2]&&LP_MANT_H[iidx][1] ? {2'b11, 1'b0} : {2'b10, LP_MANT_H[iidx][2]||LP_MANT_H[iidx][1]} ) :        ///// 5'b0111x (--> 5'b1000x)
                                LP_MANT_H[iidx][2] ? ( LP_MANT_H[iidx][1]&&LP_MANT_H[iidx][0] ? {2'b10, 1'b0} : {2'b01, LP_MANT_H[iidx][1]||LP_MANT_H[iidx][0]} ) :        ///// 5'b00111 (--> 5'b01000)
                                LP_MANT_H[iidx][1]&&LP_MANT_H[iidx][0] ? {3'b010} : {2'b00, LP_MANT_H[iidx][1]||LP_MANT_H[iidx][0]} ; // 5'b00011 (--> 5'b00100)
        assign mxfp_s1e2m1[iidx] = LP_E2M1[iidx] == 3'b000 ? 4'b0000 : {fp_sign_r[iidx], LP_E2M1[iidx]}; // remove -0.0 case
    end
endgenerate

// output
assign LP_SCALE = USE_NVFP ? scale_r : (scale_r <= `LP_EXP_MAX ? 1 : scale_r-`LP_EXP_MAX); // for MXFP, minimum scale is 1 (2^-14)
genvar lp_idx;
generate
    for (lp_idx = 0; lp_idx < GROUP_N; lp_idx = lp_idx + 1) begin : LP_DATA_GEN
        assign LP_DATA[`LP_DATAWIDTH*(lp_idx+1)-1:`LP_DATAWIDTH*lp_idx] = USE_NVFP ? nvfp_s1e2m1[lp_idx] : mxfp_s1e2m1[lp_idx];
    end
endgenerate
endmodule