`include "params.vh"


module ACT_BUFFER (
    input  clk,
    input  wen,
    input  ren,
    input  [ACT_BUFFER_ADDR_WIDTH-1:0] addr,
    input  [ACT_BUFFER_WORD_WIDTH-1:0] act_wdata,
    input  [ACT_SCALE_RF_WORD_WIDTH-1:0] act_scale_wdata,
    output [ACT_BUFFER_WORD_WIDTH-1:0] act_rdata,
    output [ACT_SCALE_RF_WORD_WIDTH-1:0] act_scale_rdata
);

    genvar i;
    generate
        for (i = 0; i < ACT_RF_BANKS; i++) begin : gen_rf_inst
            rf_sp_w160_d128 act_rf_inst (
                .CLK(clk),
                .CEN(!(ren || wen)),
                .WEN(!wen),
                .A(addr),
                .D(act_wdata[i*ACT_RF_WORD_WIDTH +: ACT_RF_WORD_WIDTH]),
                .Q(act_rdata[i*ACT_RF_WORD_WIDTH +: ACT_RF_WORD_WIDTH]),
                .EMA(3'b011),
                .EMAW(2'b01),
                .RET1N(1'b1)
            );
        end
    endgenerate

    rf_sp_w24_d128 act_scale_rf_inst (
        .CLK(clk),
        .CEN(!(ren || wen)),
        .WEN(!wen),
        .A(addr),
        .D(act_scale_wdata),
        .Q(act_scale_rdata),
        .EMA(3'b011),
        .EMAW(2'b01),
        .RET1N(1'b1)
    );


endmodule

module PSUM_BUFFER (
    input  clk,
    input  wen,
    input  ren,
    input  [PSUM_BUFFER_ADDR_WIDTH-1:0] psum_waddr,
    input  [PSUM_BUFFER_ADDR_WIDTH-1:0] psum_raddr,
    input  [PSUM_BUFFER_WORD_WIDTH-1:0] psum_wdata,
    output [PSUM_BUFFER_WORD_WIDTH-1:0] psum_rdata
);
    logic [PSUM_RF_ADDR_WIDTH-1:0] read_bank_idx, write_bank_idx;
    logic rf_bank_cen [PSUM_RF_BANKS-1:0];
    logic rf_bank_wen [PSUM_RF_BANKS-1:0];
    logic [PSUM_RF_ADDR_WIDTH-1:0] bank_addr [PSUM_RF_BANKS-1:0];
    logic [PSUM_RF_WORD_WIDTH-1:0] bank_wdata [PSUM_RF_BANKS-1:0];
    logic [PSUM_RF_WORD_WIDTH-1:0] bank_rdata [PSUM_RF_BANKS-1:0];


    always_comb begin
        // address mapping
        read_bank_idx = psum_raddr[PSUM_RF_BANK_ADDR_WIDTH-1:0]; // LSB
        write_bank_idx = psum_waddr[PSUM_RF_BANK_ADDR_WIDTH-1:0]; // LSB
        for (int i = 0; i < PSUM_RF_BANKS; i++) begin
            rf_bank_cen[i] = (ren && (read_bank_idx == i)) || (wen && (write_bank_idx == i));
            rf_bank_wen[i] = (wen && (write_bank_idx == i));
            bank_addr[i] = (wen && (write_bank_idx == i)) ? psum_waddr[PSUM_BUFFER_ADDR_WIDTH-1 -: PSUM_RF_ADDR_WIDTH] : 
                           (ren && (read_bank_idx == i)) ? psum_raddr[PSUM_BUFFER_ADDR_WIDTH-1 -: PSUM_RF_ADDR_WIDTH] : '0;
        end


        // write data mapping
        for (int i = 0; i < PSUM_RF_BANKS; i++) begin
            bank_wdata[i] = (wen && (write_bank_idx == i)) ? psum_wdata : '0;
        end
    end

    genvar i;
    generate
        for (i = 0; i < PSUM_RF_BANKS; i++) begin : gen_psum_rf_inst
            rf_sp_w136_d256 psum_rf_inst (
                .CLK(clk),
                .CEN(!rf_bank_cen[i]),
                .WEN(!rf_bank_wen[i]),
                .A(bank_addr[i]),
                .D(bank_wdata[i]),
                .Q(bank_rdata[i]),
                .EMA(3'b011),
                .EMAW(2'b01),
                .RET1N(1'b1)
            );
        end
    endgenerate

    // read data mapping
    assign psum_rdata = ren ? bank_rdata[read_bank_idx] : '0;

endmodule