`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Floating-Point to MXINT Converter                                           //
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



module MXINT8_QUANTIZER (
    input  [PSUM_WIDTH*PSUM_QUANTIZATION_B-1:0]   HP_DATA,
    output [PSUM_MX_WIDTH*PSUM_QUANTIZATION_B-1:0] LP_DATA,
    output signed [MX_SCALE_WIDTH-1:0]            LP_SCALE
);
    // -----------------------------------------------------------------------------
    // 1) Flattened HP_DATA → per-element vectors
    // -----------------------------------------------------------------------------
    wire [PSUM_WIDTH-1:0]           fp_vec      [0:PSUM_QUANTIZATION_B-1];
    wire                            fp_sign     [0:PSUM_QUANTIZATION_B-1];
    wire [PSUM_EXP_WIDTH-1:0]       fp_exp      [0:PSUM_QUANTIZATION_B-1];
    wire [PSUM_MANT_WIDTH-1:0]      fp_mant     [0:PSUM_QUANTIZATION_B-1];
    wire [PSUM_MANT_WIDTH-1:0]      fp_mant_normal     [0:PSUM_QUANTIZATION_B-1];
    wire [PSUM_EXP_WIDTH-1:0]       fp_cmp      [0:PSUM_QUANTIZATION_B-1];
    // Split packed input into parallel fields
    genvar gi;
    generate
        for (gi = 0; gi < PSUM_QUANTIZATION_B; gi = gi + 1) begin : SPLIT
            assign fp_vec[gi]  = HP_DATA[PSUM_WIDTH*(gi+1)-1:PSUM_WIDTH*gi];
            assign {fp_sign[gi], fp_exp[gi], fp_mant[gi]} = fp_vec[gi];
    //        assign fp_mant_normal[gi] = (fp_exp[gi]==0) ? 0 : {1'b1, fp_mant[gi]}; // ignore subnormal case
            // Remove sign bit, take leading MX_SCALE_WIDTH bits of magnitude for comparison
            assign fp_cmp[gi] = fp_vec[gi][PSUM_WIDTH-2 -: PSUM_EXP_WIDTH];
        end
    endgenerate
    // -----------------------------------------------------------------------------
    // 2) Shared-scale search (max of fp_cmp[])
    // -----------------------------------------------------------------------------
    localparam SEG = PSUM_QUANTIZATION_B / 8; // each MaxExpFinder consumes 8 inputs
    wire [PSUM_EXP_WIDTH-1:0] local_max [0:SEG-1];
    // Level-1 reduction: max of every 8 elements
    generate
        for (gi = 0; gi < SEG; gi = gi + 1) begin : LOCAL_MAX_GEN
            max_finder #(.BIT_WIDTH(PSUM_EXP_WIDTH)) MF (
                .data({ fp_cmp[gi*8+0], fp_cmp[gi*8+1], fp_cmp[gi*8+2], fp_cmp[gi*8+3],
                        fp_cmp[gi*8+4], fp_cmp[gi*8+5], fp_cmp[gi*8+6], fp_cmp[gi*8+7] }),
                .max (local_max[gi])
            );
        end
    endgenerate
    // Level-2 reduction: max over SEG partial maxima
    reg [PSUM_EXP_WIDTH-1:0] global_max_r;
    integer ii;
    always @* begin
        global_max_r = local_max[0];
        for (ii = 1; ii < SEG; ii = ii + 1) begin
            if (local_max[ii] > global_max_r)
                global_max_r = local_max[ii];
        end
    end
    assign LP_SCALE = global_max_r - MX_LP_EXP_MAX; // biased scale
    // -----------------------------------------------------------------------------
    // 3) Quantise mantissa to L-precision using LP_SCALE
    //    (placeholder below keeps synthesis happy)
    // -----------------------------------------------------------------------------
    genvar idx;
    wire [PSUM_EXP_WIDTH-1:0] shift_LP [0:PSUM_QUANTIZATION_B-1];
    wire  quant_to_zero [0:PSUM_QUANTIZATION_B-1];
    wire [PSUM_MANT_WIDTH+1+MX_LP_EXP_MAX-1:0] LP_MANT_H_shifted [0:PSUM_QUANTIZATION_B-1]; // +1 for leading 1, +MX_LP_EXP_MAX for exp_max
    wire [PSUM_MX_WIDTH-1:0] unsigned_lp_psum [0:PSUM_QUANTIZATION_B-1];
    wire [4:0] LP_MANT_H [0:PSUM_QUANTIZATION_B-1]; /// E2M1 pre-norm
    wire [2:0] LP_E2M1 [0:PSUM_QUANTIZATION_B-1];
    generate
        for (idx = 0; idx < PSUM_QUANTIZATION_B; idx = idx + 1) begin : sLP
            assign quant_to_zero[idx] = (fp_exp[idx] < LP_SCALE) ? 1 : 0;
            assign shift_LP[idx] = fp_exp[idx] - LP_SCALE;
            assign LP_MANT_H_shifted[idx] = (quant_to_zero[idx]) ? '0 : {{MX_LP_EXP_MAX{1'b0}}, 1'b1, fp_mant[idx]} << shift_LP[idx]; // ignore fp16 subnormal
            assign unsigned_lp_psum[idx] = LP_MANT_H_shifted[idx][PSUM_MANT_WIDTH+1+MX_LP_EXP_MAX-1 -: PSUM_MX_WIDTH];
            assign LP_DATA[PSUM_MX_WIDTH*(idx+1)-1:PSUM_MX_WIDTH*idx] = fp_sign[idx] ? ~(unsigned_lp_psum[idx]+1) : unsigned_lp_psum[idx];
        end
    endgenerate
endmodule


module MXINT8_DEQUANTIZER (
    input [PSUM_MX_WIDTH*PSUM_QUANTIZATION_B-1:0] LP_DATA,
    input signed [MX_SCALE_WIDTH-1:0]            LP_SCALE,
    output [PSUM_WIDTH*PSUM_QUANTIZATION_B-1:0]  HP_DATA
);

    logic is_zero [0:PSUM_QUANTIZATION_B-1];
    logic sign [0:PSUM_QUANTIZATION_B-1];
    logic signed [PSUM_EXP_WIDTH:0] exp [0:PSUM_QUANTIZATION_B-1];
    logic [PSUM_MANT_WIDTH-1:0] mant [0:PSUM_QUANTIZATION_B-1];
    logic [PSUM_EXP_WIDTH-1:0] scaled_exp [0:PSUM_QUANTIZATION_B-1];

    genvar b;
    generate
        for (b = 0; b < PSUM_QUANTIZATION_B; b+=1) begin : fixed2float
            FX2FP #(
                .FIXED_WIDTH(PSUM_MX_WIDTH),
                .FP_EXP_WIDTH(PSUM_EXP_WIDTH),
                .FP_MANT_WIDTH(PSUM_MANT_WIDTH)
            ) fx2fp (
                .fixed_in(LP_DATA[b*PSUM_MX_WIDTH +: PSUM_MX_WIDTH]),
                .fixed_scale(3'd0),
                .is_zero(is_zero[b]),
                .sign(sign[b]),
                .exp(exp[b]),
                .mant(mant[b])
            );
            assign scaled_exp[b] = exp[b] + LP_SCALE;
            assign HP_DATA[b*PSUM_WIDTH +: PSUM_WIDTH] = is_zero[b] ? 0 : {sign[b], scaled_exp[b], mant[b]};
        end
    endgenerate
endmodule
