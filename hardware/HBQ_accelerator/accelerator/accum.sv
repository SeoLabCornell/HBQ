`include "params.vh";


module ACCUM (
    input clk,
    input reset,
    input drain,
    input wen,
    input [16*SA_N-1:0] scaled_psum,
    output reg result_val,
    output reg [16*SA_N-1:0] result
);
    logic drain_mode_r, drain_mode_w; // 1 for drain, 0 for normal
    logic buffer_full_r, buffer_full_w;
    // logic psum_wen_r, psum_wen_w;
    // logic psum_ren_r, psum_ren_w;
    logic [PSUM_BUFFER_ADDR_WIDTH-1:0] read_ptr_r, read_ptr_w;
    logic [PSUM_BUFFER_ADDR_WIDTH-1:0] write_ptr_r, write_ptr_w;
    logic [PSUM_WIDTH*SA_N-1:0] fp16_psum;
    logic wen_r;
    logic [16*SA_N-1:0] scaled_psum_r;
    logic [PSUM_BUFFER_WORD_WIDTH-1:0] psum_rdata;
    logic [PSUM_BUFFER_WORD_WIDTH-1:0] psum_buffer_word_r, psum_buffer_word_w;
    logic [PSUM_WIDTH*SA_N-1:0] dequant_fp16_psum;

    assign buffer_full_w = buffer_full_r || (write_ptr_r == (PSUM_BUFFER_DEPTH-1));
    always_comb begin
        if (drain) begin
            drain_mode_w = 1;
        end
        else if (drain_mode_r && read_ptr_r != (PSUM_BUFFER_DEPTH-1)) begin
            drain_mode_w = 1;
        end
        else begin
            drain_mode_w = 0;
        end

        if (drain_mode_r) read_ptr_w = read_ptr_r+1;
        else if (wen) read_ptr_w = read_ptr_r+1;
        else read_ptr_w = read_ptr_r;

        if (wen_r) write_ptr_w = write_ptr_r + 1;
        else write_ptr_w = write_ptr_r;

        if (drain_mode_r) begin
            result_val = 1;
            result = dequant_fp16_psum;
        end
        else begin
            result_val = 0;
            result = 0;
        end
    end

    PSUM_BUFFER psum_buffer (
        .clk(clk),
        .wen(wen_r),
        .ren(drain || drain_mode_r || wen),
        .psum_waddr(write_ptr_r),
        .psum_raddr(read_ptr_r),
        .psum_wdata(psum_buffer_word_r),
        .psum_rdata(psum_rdata)
    );

    // dequantization
    
    MXINT8_DEQUANTIZER dequant (
        .LP_DATA(psum_rdata[PSUM_BUFFER_WORD_WIDTH-1 -: PSUM_MX_WIDTH*SA_N]),
        .LP_SCALE(psum_rdata[PSUM_BUFFER_WORD_WIDTH-PSUM_MX_WIDTH*SA_N-1 -: MX_SCALE_WIDTH]),
        .HP_DATA(dequant_fp16_psum)
    );

    genvar i;
    generate
        for (i = 0; i < SA_N; i++) begin
            FP16_ADDER fp16_adder (
                .ADD2SUB('0),
                .IN1(scaled_psum_r[PSUM_WIDTH*i +: PSUM_WIDTH]),
                .IN2(!buffer_full_r ? '0 : dequant_fp16_psum[PSUM_WIDTH*i +: PSUM_WIDTH]),
                .OUT(fp16_psum[PSUM_WIDTH*i +: PSUM_WIDTH])
            );
        end
    endgenerate

    // quantization
    
    MXINT8_QUANTIZER quant (
        .HP_DATA(fp16_psum),
        .LP_DATA(psum_buffer_word_w[PSUM_BUFFER_WORD_WIDTH-1 -: PSUM_MX_WIDTH*SA_N]),
        .LP_SCALE(psum_buffer_word_w[PSUM_BUFFER_WORD_WIDTH-PSUM_MX_WIDTH*SA_N-1 -: MX_SCALE_WIDTH])
    );


    

    always_ff @(posedge clk) begin
        if (reset) begin
            buffer_full_r <= 0;
            read_ptr_r <= 0;
            write_ptr_r <= 0;
            psum_buffer_word_r <= 0;
            drain_mode_r <= 0;
            wen_r <= 0;
            scaled_psum_r <= 0;
        end
        else begin
            buffer_full_r <= buffer_full_w;
            read_ptr_r <= read_ptr_w;
            write_ptr_r <= write_ptr_w;
            psum_buffer_word_r <= psum_buffer_word_w;
            drain_mode_r <= drain_mode_w;
            wen_r <= wen;
            scaled_psum_r <= scaled_psum;
        end
    end

endmodule