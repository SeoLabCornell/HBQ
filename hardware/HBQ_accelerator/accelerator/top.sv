`include "params.vh";


module TOP (
    input  clk,
    input  reset_accum,
    input  drain,
    input  [2:0] mode,
    input  propagate_wgt,
    input  act_wen,
    input  act_ren,
    input  [ACT_BUFFER_ADDR_WIDTH-1:0] act_addr,
    input  [ACT_BUFFER_WORD_WIDTH-1:0] act_wdata,
    input  [ACT_SCALE_RF_WORD_WIDTH-1:0] act_scale_wdata,
    input  [X*B-1:0]     wgt_vec,
    input  [7:0]  scale_w,
    input  [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input  l2_scale_w_scheme,
    output result_val,
    output [16*SA_N-1:0] result
);
    // SA signals
    logic [16*SA_N-1:0] scaled_psum_out;
    logic [7:0] scale_x_r, scale_x_rr;
    logic [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x_r;


    // output buffer signals
    logic wen_r [0:SA_PP_STAGE-1];

    // ACT buffer signals
    logic [ACT_BUFFER_WORD_WIDTH-1:0] act_rdata;
    logic [ACT_SCALE_RF_WORD_WIDTH-1:0] act_scale_rdata;

    // control register
    logic drain_r;
    logic [2:0] mode_r;

    // Psum buffer signals
    logic psum_w_en, psum_r_en;
    logic [PSUM_BUFFER_ADDR_WIDTH-1:0] psum_waddr;
    logic [PSUM_BUFFER_ADDR_WIDTH-1:0] psum_raddr;
    logic [PSUM_BUFFER_WORD_WIDTH-1:0] psum_wdata;
    logic [PSUM_BUFFER_WORD_WIDTH-1:0] psum_rdata;


	reg [ACT_BUFFER_WORD_WIDTH-1:0] act_rdata_FF;
    // SA signals
    SA systolic_array (
        .clk(clk),
        .reset_accum(reset_accum),
        .mode(mode_r),
        .propagate_wgt(propagate_wgt),
        .in_val(wen_r[0]),
        .act_vec(act_rdata_FF),
        .wgt_vec(wgt_vec),
        .scale_x(scale_x_rr), // delay two cycles
        .scale_w(scale_w),
        .l2_scale_x(l2_scale_x_r), // delay one cycle
        .l2_scale_w(l2_scale_w),
        .l2_scale_w_scheme(l2_scale_w_scheme),
        .scaled_psum_out(scaled_psum_out)
    );

    ACT_BUFFER act_buffer (
        .clk(clk),
        .wen(act_wen),
        .ren(act_ren),
        .addr(act_addr),
        .act_wdata(act_wdata),
        .act_scale_wdata(act_scale_wdata),
        .act_rdata(act_rdata),
        .act_scale_rdata(act_scale_rdata)
    );

    // wait for psum comes out
    
    ACCUM accum_unit (
        .clk(clk),
        .reset(reset_accum),
        .drain(drain_r),
        .wen(wen_r[SA_PP_STAGE-1]),
        .scaled_psum(scaled_psum_out),
        .result_val(result_val),
        .result(result)
    );

    always_ff @(posedge clk) begin
        wen_r[0] <= act_ren;
        mode_r <= mode;
        drain_r <= drain;
		act_rdata_FF <= act_rdata;
        for (int i = 0; i < SA_PP_STAGE-1; i+=1) begin
            wen_r[i+1] <= wen_r[i];
        end
        if (wen_r[0]) begin
            l2_scale_x_r <= act_scale_rdata[8 +: L2_SCALE_X_BIT*NUM_SUB_B];
            scale_x_r <= act_scale_rdata[0 +: 8];
        end
        else begin
            l2_scale_x_r <= 0;
            scale_x_r <= 0;
        end
        scale_x_rr <= scale_x_r;
    end

endmodule