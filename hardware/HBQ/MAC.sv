`include "params.vh";

module WXAY_MAC_NV (
    input                     clk,
    input                     reset_accum,
    input                     drain,
    input  [Y*B-1:0]          act_vec,
    input  [X*B-1:0]          wgt_vec,
    input  [7:0]              scale_x,
    input  [7:0]              scale_w,
    input  [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x,
    input  [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input                     l2_scale_w_scheme,
    input  [15:0] fp16_psum_in,

    output reg                drain_out,
    output reg [Y*B-1:0]      act_vec_out,
    output reg [X*B-1:0]      wgt_vec_out,
    output [7:0]              scale_x_out,
    output [7:0]              scale_w_out,
    output [L2_SCALE_X_BIT*NUM_SUB_B-1:0]  l2_scale_x_out,
    output [L2_SCALE_W_BIT*NUM_SUB_B-1:0]  l2_scale_w_out,
    output                    l2_scale_w_scheme_out,
    output [15:0] fp16_psum_out,

    output [PSUM_WIDTH-1:0] psum_debug,
    output [15:0] scaled_psum_debug,
    output [L2_PSUM_WIDTH*NUM_SUB_B-1:0] l2_psum_debug
);
    // localparam MODE_128 = 1;
    // localparam MODE_L2  = 2;

    wire [PSUM_WIDTH-1:0] psum;
    wire [15:0] scaled_psum;

    SYS_REG #(
        .X(X),
        .Y(Y),
        .B(B),
        .NUM_SUB_B(NUM_SUB_B)
        // .L2_SCALE_BIT(L2_SCALE_BIT)
    ) systolic_reg (
        .clk(clk),
        .drain(drain),
        .act_vec(act_vec),
        .wgt_vec(wgt_vec),
        .scale_x(scale_x),
        .scale_w(scale_w),
        .l2_scale_x(l2_scale_x),
        .l2_scale_w(l2_scale_w),
        .l2_scale_w_scheme(l2_scale_w_scheme),
        .drain_out(drain_out),
        .act_vec_out(act_vec_out),
        .wgt_vec_out(wgt_vec_out),
        .scale_x_out(scale_x_out),
        .scale_w_out(scale_w_out),
        .l2_scale_x_out(l2_scale_x_out),
        .l2_scale_w_out(l2_scale_w_out),
        .l2_scale_w_scheme_out(l2_scale_w_scheme_out)
    );

    FP_WXAY_vector mac_vec (
        .clk(clk),
        .ACT(act_vec),
        .WEIGHT(wgt_vec),
        .l2_scale_w(l2_scale_w),
        .l2_scale_x(l2_scale_x),
        .l2_scale_w_scheme(l2_scale_w_scheme),
        .PSUM(psum),
        .l2_psum_debug(l2_psum_debug)
    );
    assign psum_debug = psum;
    
    reg [7:0] scale_x_in; // slicing scale input
    reg [7:0] scale_w_in; // slicing scale input
    reg [7:0] scale_x_dequant; // input of dequantization
    reg [7:0] scale_w_dequant; // input of dequantization
    reg [2:0] fixed_scale;

    // Scaling factor
    always_comb begin
        scale_x_in = scale_x;
        scale_w_in = scale_w;
        scale_x_dequant = scale_x_in;
        scale_w_dequant = scale_w_in;
        fixed_scale = X+Y-6;
    end

    FX2FP_Scale_NV #(
        .FIXED_WIDTH(PSUM_WIDTH) // add 2 more level to support b=64&128
    ) dequant_0 (
        .clk(clk),
        .clk_gate(1'b0),
        .fixed_scale(fixed_scale),
        .psum_fixed_in(psum),
        .scale_x(scale_x_dequant),
        .scale_w(scale_w_dequant),
        .float_out(scaled_psum)
    );
    assign scaled_psum_debug = scaled_psum;

    FP16_ACCUM fp16_accum_0 (
        .clk(clk),
        .drain(drain),
        .clk_gate(1'b0),
        .reset_accum(reset_accum),
        .psum_prev(fp16_psum_in),
        .scaled_psum(scaled_psum),
        .psum_out(fp16_psum_out)
    );


endmodule

module SYS_REG #(
    parameter integer X = 4,  // 3, 4, 5, 6, 7, 8
    parameter integer Y = 5,  // 4, 5, 6, 7, 8
    parameter integer B = 128, // 4, 8, 16, 32, 64, 128
    parameter integer NUM_SUB_B = 4
) (
    input                     clk,
    input                     drain,
    input  [Y*B-1:0]          act_vec,
    input  [X*B-1:0]          wgt_vec,
    input  [7:0]  scale_x,
    input  [7:0]  scale_w,
    input  [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x,
    input  [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input  l2_scale_w_scheme,

    output reg                   drain_out,
    output reg [Y*B-1:0]         act_vec_out,
    output reg [X*B-1:0]         wgt_vec_out,
    output reg [7:0] scale_x_out,
    output reg [7:0] scale_w_out,
    output reg [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x_out,
    output reg [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w_out,
    output reg l2_scale_w_scheme_out
);
    always @(posedge clk) begin
        act_vec_out <= act_vec;
        wgt_vec_out <= wgt_vec;
        scale_x_out <= scale_x;
        scale_w_out <= scale_w;
        l2_scale_x_out <= l2_scale_x;
        l2_scale_w_out <= l2_scale_w;
        l2_scale_w_scheme_out <= l2_scale_w_scheme;
        drain_out   <= drain;
    end
endmodule
