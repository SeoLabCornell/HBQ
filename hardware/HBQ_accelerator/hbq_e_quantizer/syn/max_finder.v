`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
module max_finder #(parameter BIT_WIDTH=6)
(
    input [8*BIT_WIDTH-1:0]  data,
    output reg [BIT_WIDTH-1:0] max
);
    wire [BIT_WIDTH-1:0] data_in [0:7];
    genvar idx;
    generate
        for (idx = 0; idx < 8; idx = idx + 1) begin : ig
            assign data_in[idx] = data[BIT_WIDTH*(idx+1)-1 -: BIT_WIDTH];
        end
    endgenerate
    reg [BIT_WIDTH-1:0] temp1_exp [0:3];
    reg [BIT_WIDTH-1:0] temp2_exp [0:1];
    always @ (*) begin
        temp1_exp[0] = (data_in[0] > data_in[1]) ? data_in[0] : data_in[1];
        temp1_exp[1] = (data_in[2] > data_in[3]) ? data_in[2] : data_in[3];
        temp1_exp[2] = (data_in[4] > data_in[5]) ? data_in[4] : data_in[5];
        temp1_exp[3] = (data_in[6] > data_in[7]) ? data_in[6] : data_in[7];
        temp2_exp[0] = (temp1_exp[0] > temp1_exp[1]) ? temp1_exp[0] : temp1_exp[1];
        temp2_exp[1] = (temp1_exp[2] > temp1_exp[3]) ? temp1_exp[2] : temp1_exp[3];
        max = (temp2_exp[0] > temp2_exp[1]) ? temp2_exp[0] : temp2_exp[1] ;
    end
endmodule

module max_finder_16_input #(parameter BIT_WIDTH=6)
(
    input [16*BIT_WIDTH-1:0]  data,
    output reg [BIT_WIDTH-1:0] max
);
    wire [BIT_WIDTH-1:0] data_in [0:15];
    genvar idx;
    generate
        for (idx = 0; idx < 16; idx = idx + 1) begin : ig
            assign data_in[idx] = data[BIT_WIDTH*(idx+1)-1 -: BIT_WIDTH];
        end
    endgenerate
    reg [BIT_WIDTH-1:0] temp1_exp [0:7];
    reg [BIT_WIDTH-1:0] temp2_exp [0:3];
    reg [BIT_WIDTH-1:0] temp3_exp [0:1];
    always @ (*) begin
        temp1_exp[0] = (data_in[0] > data_in[1]) ? data_in[0] : data_in[1];
        temp1_exp[1] = (data_in[2] > data_in[3]) ? data_in[2] : data_in[3];
        temp1_exp[2] = (data_in[4] > data_in[5]) ? data_in[4] : data_in[5];
        temp1_exp[3] = (data_in[6] > data_in[7]) ? data_in[6] : data_in[7];
        temp1_exp[4] = (data_in[8] > data_in[9]) ? data_in[8] : data_in[9];
        temp1_exp[5] = (data_in[10] > data_in[11]) ? data_in[10] : data_in[11];
        temp1_exp[6] = (data_in[12] > data_in[13]) ? data_in[12] : data_in[13];
        temp1_exp[7] = (data_in[14] > data_in[15]) ? data_in[14] : data_in[15];
        temp2_exp[0] = (temp1_exp[0] > temp1_exp[1]) ? temp1_exp[0] : temp1_exp[1];
        temp2_exp[1] = (temp1_exp[2] > temp1_exp[3]) ? temp1_exp[2] : temp1_exp[3];
        temp2_exp[2] = (temp1_exp[4] > temp1_exp[5]) ? temp1_exp[4] : temp1_exp[5];
        temp2_exp[3] = (temp1_exp[6] > temp1_exp[7]) ? temp1_exp[6] : temp1_exp[7];
        temp3_exp[0] = (temp2_exp[0] > temp2_exp[1]) ? temp2_exp[0] : temp2_exp[1];
        temp3_exp[1] = (temp2_exp[2] > temp2_exp[3]) ? temp2_exp[2] : temp2_exp[3];
        max = (temp3_exp[0] > temp3_exp[1]) ? temp3_exp[0] : temp3_exp[1] ;
    end
endmodule