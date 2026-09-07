
`include "params.vh";

module WXAY_MAC_NV (
    input                     clk,
    input                     reset_accum,
    input  [2:0]              mode,
    input                     propagate_wgt,
    input  [Y*B-1:0]          act_vec,
    input  [X*B-1:0]          wgt_vec,
    input  [7:0]              scale_x,
    input  [7:0]              scale_w,
    input  [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x,
    input  [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input                     l2_scale_w_scheme,

    output reg                propagate_wgt_out,
    output reg [X*B-1:0]      wgt_vec_out,
    output [7:0]              scale_w_out,
    output [L2_SCALE_W_BIT*NUM_SUB_B-1:0]  l2_scale_w_out,
    output                    l2_scale_w_scheme_out,
    // output [15:0]             fp16_psum_out,
    output [15:0]             scaled_psum_out

);
    logic [L2_PSUM_WIDTH*NUM_SUB_B-1:0] l2_psum_debug;

    wire [FX_PSUM_WIDTH-1:0] psum;
    wire [15:0] scaled_psum;

    SYS_REG #(
        .X(X),
        .Y(Y),
        .B(B),
        .NUM_SUB_B(NUM_SUB_B)
        // .L2_SCALE_BIT(L2_SCALE_BIT)
    ) systolic_reg (
        .clk(clk),
        .propagate_wgt(propagate_wgt),
        .wgt_vec(wgt_vec),
        .scale_w(scale_w),
        .l2_scale_w(l2_scale_w),
        .l2_scale_w_scheme(l2_scale_w_scheme),
        .propagate_wgt_out(propagate_wgt_out),
        .wgt_vec_out(wgt_vec_out),
        .scale_w_out(scale_w_out),
        .l2_scale_w_out(l2_scale_w_out),
        .l2_scale_w_scheme_out(l2_scale_w_scheme_out)
    );

    FP_WXAY_vector mac_vec (
        .clk(clk),
        .mode(mode),
        .ACT(act_vec),
        .WEIGHT(wgt_vec_out),
        .l2_scale_w(l2_scale_w_out),
        .l2_scale_x(l2_scale_x),
        .l2_scale_w_scheme(l2_scale_w_scheme_out),
        .PSUM(psum),
        .l2_psum_debug(l2_psum_debug)
    );
    // assign psum_debug = psum;
    
    reg [2:0] fixed_scale;
    // Scaling factor
    always_comb begin
        fixed_scale = 3;
        case (mode) // use pipelined mode
            MODE_128: begin
                fixed_scale = 3;
            end
            MODE_L2: begin
                fixed_scale = 6;
            end
        endcase
    end
    FX2FP_Scale_NV #(
        .FIXED_WIDTH(FX_PSUM_WIDTH)
    ) dequant_0 (
        .clk(clk),
        .clk_gate(1'b0),
        .fixed_scale(fixed_scale),
        .psum_fixed_in(psum),
        .scale_x(scale_x),
        .scale_w(scale_w_out),
        .float_out(scaled_psum)
    );
    assign scaled_psum_out = scaled_psum;

endmodule

module SYS_REG #(
    parameter integer X = 4,  // 3, 4, 5, 6, 7, 8
    parameter integer Y = 5,  // 4, 5, 6, 7, 8
    parameter integer B = 128, // 4, 8, 16, 32, 64, 128
    parameter integer NUM_SUB_B = 4
) (
    input                     clk,
    input                     propagate_wgt,
    input  [X*B-1:0]          wgt_vec,
    input  [7:0]              scale_w,
    input  [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input  l2_scale_w_scheme,

    output reg propagate_wgt_out,
    output reg [X*B-1:0]         wgt_vec_out,
    output reg [7:0] scale_w_out,
    output reg [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w_out,
    output reg l2_scale_w_scheme_out
);
    always @(posedge clk) begin
        if (propagate_wgt) begin
            wgt_vec_out <= wgt_vec;
            scale_w_out <= scale_w;
            l2_scale_w_out <= l2_scale_w;
            l2_scale_w_scheme_out <= l2_scale_w_scheme;
            propagate_wgt_out <= propagate_wgt;
        end
        // else begin
        //     wgt_vec_out <= wgt_vec_out;
        //     scale_w_out <= scale_w_out;
        //     l2_scale_w_out <= l2_scale_w_out;
        //     l2_scale_w_scheme_out <= l2_scale_w_scheme_out;
        //     propagate_wgt_out <= propagate_wgt_out;
        // end
    end
endmodule
