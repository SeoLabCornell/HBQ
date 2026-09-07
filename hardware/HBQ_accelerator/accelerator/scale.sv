/*
-----------------------------------------------------------------------------
Module: FX2FP_Scale_NV
    Convert fixed point input (adder tree output) to FP16, and scale with activation and weight scaling factors
    Support both MX and NV format
    Input:
        psum_fixed_in: signed fixed point partial sum (implicit scale 2^-FIXED_SCALE) (will be one cycle delayed compared to scale_x & scale_w)
        scale_x: 8b scale for activation
        scale_w: 8b scale for weight
    Output:
        float_out: 16b floating point output (S1E5M10)
Module: FX2FP
    Convert fixed point into FP format
Module: Scale
    Scale partial sum with scaling factor input
Module: Scale_NV
    Does same thing as Scale, but only support NV mode
Module: Scale_MX
    Does same thing as Scale, but only support MX mode
*/

`timescale 1ns / 1ps

module FX2FP #(
    parameter FIXED_WIDTH = 13,
    parameter FP_EXP_WIDTH = 5,
    parameter FP_MANT_WIDTH = 10
) (
    input signed [FIXED_WIDTH-1:0] fixed_in,
    input [2:0] fixed_scale,
    output reg is_zero,
    output reg sign,
    output reg signed [FP_EXP_WIDTH:0] exp,
    output reg [FP_MANT_WIDTH-1:0] mant
);
    reg [FIXED_WIDTH-1:0]  abs_val;
    reg [4:0]   leading_zeros;
    integer     i;
    reg         found;

    // Extra wide register for normalized value (to safely hold shifted result)
    reg [FIXED_WIDTH-1:0]  normalized;
    always @(*) begin
        sign = fixed_in[FIXED_WIDTH-1];
        if (sign)
            abs_val = ~fixed_in + 1;  // 2's complement conversion for negative numbers
        else
            abs_val = fixed_in;       
        found = 1'b0;
        leading_zeros = 0;
        for (i = FIXED_WIDTH-1; i >= 0; i = i - 1) begin
            if (!found && abs_val[i]) begin
                leading_zeros = FIXED_WIDTH - i - 1;
                found = 1'b1;
            end
        end
        exp = FIXED_WIDTH - fixed_scale - leading_zeros - 1;
        normalized = abs_val << leading_zeros;
        if (FP_MANT_WIDTH >= FIXED_WIDTH-2) begin
            mant = normalized[FIXED_WIDTH-2:0];
        end
        else begin
            mant = normalized[FIXED_WIDTH-2-:FP_MANT_WIDTH]; // -1 for leading 1
        end
        is_zero = (abs_val == 0);
    end
endmodule

module FX2FP_Scale_NV #(
    parameter FIXED_WIDTH = 13,
    parameter SCALE_EXP_BIAS = 15,
    parameter FP_EXP_MAX_UNBIASED = 30, // 30 = 15 + 15
    parameter FP_EXP_WIDTH = 5,
    parameter FP_MANT_WIDTH = 10
) (
    input clk,
    input clk_gate,
    input [2:0] fixed_scale,
    input signed [FIXED_WIDTH-1:0] psum_fixed_in,
    input [7:0] scale_x,
    input [7:0] scale_w,
    output reg [15:0] float_out
);
    // Convert fixed point input to FP16
    wire psum_is_zero;
    wire psum_sign;
    wire [FP_EXP_WIDTH:0] psum_exp;
    wire [FP_MANT_WIDTH-1:0] psum_mant;
    FX2FP #(
        .FIXED_WIDTH(FIXED_WIDTH),
        .FP_EXP_WIDTH(FP_EXP_WIDTH),
        .FP_MANT_WIDTH(FP_MANT_WIDTH)
    ) fx2fp (
        .fixed_in(psum_fixed_in),
        .fixed_scale(fixed_scale),
        .is_zero(psum_is_zero),
        .sign(psum_sign),
        .exp(psum_exp),
        .mant(psum_mant)
    );

    // Scale
    reg [15:0] scaled_psum;
    Scale_NV #(
        .FIXED_WIDTH(FIXED_WIDTH),
        .FP_EXP_MAX_UNBIASED(FP_EXP_MAX_UNBIASED)
    ) scale (
        .psum_is_zero(psum_is_zero),
        .psum_sign(psum_sign),
        .psum_exp(psum_exp),
        .psum_mant(psum_mant),
        .scale_w(scale_w),
        .scale_x(scale_x),
        .float_out(scaled_psum)
    );
    always @(posedge clk) begin
        if (!clk_gate) begin
            float_out <= scaled_psum;
        end
    end
endmodule

module Scale_NV #(
    parameter FIXED_WIDTH = 13,
    parameter SCALE_EXP_BIAS = 15,
    parameter FP_EXP_MAX_UNBIASED = 30, // 30 = 15 + 15
    parameter FP_EXP_WIDTH = 5,
    parameter FP_MANT_WIDTH = 10
) (
    input psum_is_zero,
    input psum_sign,
    input signed [FP_EXP_WIDTH:0] psum_exp,
    input [FP_MANT_WIDTH-1:0] psum_mant,
    input [7:0] scale_x,
    input [7:0] scale_w,
    output reg [15:0] float_out
);
    reg signed [5:0] scale_x_exp, scale_w_exp;
    reg [2:0] nv_scale_x_mant, nv_scale_w_mant;
    reg [7:0] nv_scale_mant_product;
    reg signed [5:0] exp_sum; // add one bit for overflow
    reg [18:0] product;
    reg signed [FP_EXP_WIDTH:0] fp_exp; // add one bit for overflow
    reg [FP_MANT_WIDTH-1:0] fp_mant;
    reg round;
    always @(*) begin
        scale_x_exp = {1'b0, scale_x[7:3]}; // biased
        scale_w_exp = {1'b0, scale_w[7:3]}; // biased
        nv_scale_x_mant = scale_x[2:0];
        nv_scale_w_mant = scale_w[2:0];
        exp_sum = scale_x_exp + scale_w_exp + psum_exp - SCALE_EXP_BIAS;
        nv_scale_mant_product = {1'b1, nv_scale_x_mant} * {1'b1, nv_scale_w_mant}; // 1x.xxxxxx or 01.xxxxxx
        product = {1'b1, psum_mant} * nv_scale_mant_product; // 1xx.10b or 01x.10b or 001.10b
        if (product[18]) begin // 1xx.10b
            fp_exp = exp_sum+2;
            fp_mant = product[17-:FP_MANT_WIDTH];
            round = product[17-FP_MANT_WIDTH];
        end else if (product[17]) begin // 01x.10b
            fp_exp = exp_sum+1;
            fp_mant = product[16-:FP_MANT_WIDTH];
            round = product[16-FP_MANT_WIDTH];
        end else begin // 001.10b
            fp_exp = exp_sum;
            fp_mant = product[15-:FP_MANT_WIDTH];
            round = product[15-FP_MANT_WIDTH];
        end
        if (round) begin
            if (fp_mant == {FP_MANT_WIDTH{1'b1}}) begin
                fp_exp = fp_exp + 1;
                fp_mant = 0;
            end else begin
                fp_mant = fp_mant + 1;
            end
        end

        if (psum_is_zero) begin
            float_out = {1'b0, {5'b00000}, {10'b0000000000}};
        end else if (fp_exp > FP_EXP_MAX_UNBIASED) begin // overflow, clip to max
            float_out = {psum_sign, {5'b11110}, {10'b1111111111}};
        end else if (fp_exp < 1) begin // underflow, clip to 0
            float_out = {1'b0, {5'b00000}, {10'b0000000000}};
        end else begin
            float_out = {psum_sign, fp_exp[FP_EXP_WIDTH-1:0], fp_mant};
        end
    end
endmodule