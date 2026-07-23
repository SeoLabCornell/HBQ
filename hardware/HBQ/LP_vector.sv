/*
    Low precision Vector MAC unit
    Modules:
        FP_WXAY_vector: Parameterized design for WXAYB design
        FP_mul_vector: Parameterized design for E2 FP multiplier, with converter to fixed point
        ADDER_TREE_{B}: Adder tree module for different block size
*/

`include "params.vh"

module FP_WXAY_vector (
    input  wire                          clk,
    input  wire [X*B-1:0]                WEIGHT,
    input  wire [Y*B-1:0]                ACT,
    input  wire [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input  wire [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x,
    input  wire                          l2_scale_w_scheme,
    output reg  [PSUM_WIDTH-1:0] PSUM,
    output reg  [L2_PSUM_WIDTH*NUM_SUB_B-1:0] l2_psum_debug
);

    logic [X*SUB_B-1:0]             wgt_sub_block[0:NUM_SUB_B];
    logic [Y*SUB_B-1:0]             act_sub_block[0:NUM_SUB_B];
    logic [PRODUCT_WIDTH*SUB_B-1:0] product[0:NUM_SUB_B-1];
    logic signed [L2_PSUM_WIDTH-1:0] sub_block_psum[0:NUM_SUB_B-1];
    logic signed [L2_PSUM_WIDTH-1:0] sub_block_psum_r[0:NUM_SUB_B-1];
    logic signed [SCALED_L2_PSUM_WIDTH*NUM_SUB_B-1:0] scaled_l2_psum;
    logic signed [PSUM_WIDTH-1:0] psum;


    genvar sub_block;
    generate
        for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : SUB_B_IN
            assign wgt_sub_block[sub_block] = WEIGHT[X*SUB_B*sub_block +: X*SUB_B];
            assign act_sub_block[sub_block] = ACT[Y*SUB_B*sub_block +: Y*SUB_B];
        end 
    endgenerate

    generate
        for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : FP_MUL_VEC
            FP_mul_vector #(.N(SUB_B), .X(X), .Y(Y)) mul_vec (
                .WEIGHT(wgt_sub_block[sub_block]),
                .ACT(act_sub_block[sub_block]),
                .PRODUCT(product[sub_block])
            );
        end
    endgenerate

    generate
        if (SUB_B == 64) begin
            for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : ADDER_TREE
                ADDER_TREE_64 #(.N(SUB_B), .LEVELS(SUB_BLOCK_LEVEL), .WIDTH(PRODUCT_WIDTH)) adder_tree (
                    .fixed_point_in(product[sub_block]),
                    .sum(sub_block_psum[sub_block])
                );
                assign l2_psum_debug[sub_block*L2_PSUM_WIDTH +: L2_PSUM_WIDTH] = sub_block_psum_r[sub_block];
            end
        end
        else if (SUB_B == 32) begin
            for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : ADDER_TREE
                ADDER_TREE_32 #(.N(SUB_B), .LEVELS(SUB_BLOCK_LEVEL), .WIDTH(PRODUCT_WIDTH)) adder_tree (
                    .fixed_point_in(product[sub_block]),
                    .sum(sub_block_psum[sub_block])
                );
                assign l2_psum_debug[sub_block*L2_PSUM_WIDTH +: L2_PSUM_WIDTH] = sub_block_psum_r[sub_block];
            end
        end
        else if (SUB_B == 16) begin
            for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : ADDER_TREE
                ADDER_TREE_16 #(.N(SUB_B), .LEVELS(SUB_BLOCK_LEVEL), .WIDTH(PRODUCT_WIDTH)) adder_tree (
                    .fixed_point_in(product[sub_block]),
                    .sum(sub_block_psum[sub_block])
                );
                assign l2_psum_debug[sub_block*L2_PSUM_WIDTH +: L2_PSUM_WIDTH] = sub_block_psum_r[sub_block];
            end
        end
        else if (SUB_B == 8) begin
            for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : ADDER_TREE
                ADDER_TREE_8 #(.N(SUB_B), .LEVELS(SUB_BLOCK_LEVEL), .WIDTH(PRODUCT_WIDTH)) adder_tree (
                    .fixed_point_in(product[sub_block]),
                    .sum(sub_block_psum[sub_block])
                );
                assign l2_psum_debug[sub_block*L2_PSUM_WIDTH +: L2_PSUM_WIDTH] = sub_block_psum_r[sub_block];
            end
        end
        else if (SUB_B == 4) begin
            for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : ADDER_TREE
                ADDER_TREE_4 #(.N(SUB_B), .LEVELS(SUB_BLOCK_LEVEL), .WIDTH(PRODUCT_WIDTH)) adder_tree (
                    .fixed_point_in(product[sub_block]),
                    .sum(sub_block_psum[sub_block])
                );
                assign l2_psum_debug[sub_block*L2_PSUM_WIDTH +: L2_PSUM_WIDTH] = sub_block_psum_r[sub_block];
            end
        end
        else if (SUB_B == 2) begin
            for (sub_block = 0; sub_block < NUM_SUB_B; sub_block = sub_block+1) begin : ADDER_TREE
                ADDER_TREE_2 #(.N(SUB_B), .LEVELS(SUB_BLOCK_LEVEL), .WIDTH(PRODUCT_WIDTH)) adder_tree (
                    .fixed_point_in(product[sub_block]),
                    .sum(sub_block_psum[sub_block])
                );
                assign l2_psum_debug[sub_block*L2_PSUM_WIDTH +: L2_PSUM_WIDTH] = sub_block_psum_r[sub_block];
            end
        end
    endgenerate



    // L2 scale
    logic signed [3:0] l2_a_dec_scale [0:NUM_SUB_B-1];
    logic signed [4:0] l2_w_dec_scale [0:NUM_SUB_B-1];
    logic signed [L2_PSUM_WIDTH+3:0] sub_block_psum_scaled_x[0:NUM_SUB_B-1];
    always_comb begin
        for (int sub_b = 0; sub_b < NUM_SUB_B; sub_b+=1) begin
            // sub_block_psum_scaled_x[sub_b] = 0;
            // l2_scale_product[sub_b] = (l2_scale_w_scheme == L2_SCHEME_FP) ? 
            //     (l2_scale_x[sub_b*L2_SCALE_X_BIT +: L2_SCALE_X_BIT]+2) * {1'b1, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT], 1'b0} :
            //     (l2_scale_x[sub_b*L2_SCALE_X_BIT +: L2_SCALE_X_BIT]+2) * {2'b10, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT]};
            l2_a_dec_scale[sub_b] = $signed({1'b0, (l2_scale_x[sub_b*L2_SCALE_X_BIT +: L2_SCALE_X_BIT]+2)});
            l2_w_dec_scale[sub_b] = (l2_scale_w_scheme == L2_SCHEME_FP) ? {2'b01, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT], 1'b0} : {3'b010, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT]};
            sub_block_psum_scaled_x[sub_b] = sub_block_psum_r[sub_b][L2_PSUM_WIDTH-1:1] * l2_a_dec_scale[sub_b]; // truncate 1-bit for l2-a-scale
            scaled_l2_psum[sub_b*SCALED_L2_PSUM_WIDTH +: SCALED_L2_PSUM_WIDTH] = sub_block_psum_scaled_x[sub_b][L2_PSUM_WIDTH+3 : 3] * l2_w_dec_scale[sub_b]; // truncate 3-bit for 1.xxx in l2-w-scale

            // if (L2_SCALE_X_SCHEME == 0) begin // PoT
            //     sub_block_psum_scaled_x[sub_b] = sub_block_psum_r[sub_b] << l2_scale_x[sub_b*L2_SCALE_X_BIT +: L2_SCALE_X_BIT];
            // end
            // else if (L2_SCALE_X_SCHEME == 1) begin // INT
            //     sub_block_psum_scaled_x[sub_b] = sub_block_psum_r[sub_b] * $signed({1'b0, l2_scale_x[sub_b*L2_SCALE_X_BIT +: L2_SCALE_X_BIT]+1});
            // end
            // else if (L2_SCALE_X_SCHEME == 2) begin // SIG-1
            //     sub_block_psum_scaled_x[sub_b] = (sub_block_psum_r[sub_b]>>>1) * $signed({1'b0, l2_scale_x[sub_b*L2_SCALE_X_BIT +: L2_SCALE_X_BIT]+2});
            // end

            // 2-bit int L2-scale for weight
            // if (L2_SCALE_W_SCHEME == 0) begin // INT
            //     scaled_l2_psum[sub_b*SCALED_L2_PSUM_WIDTH +: SCALED_L2_PSUM_WIDTH] = (sub_block_psum_scaled_x[sub_b] * $signed({2'b01, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT], 1'b0}));
            // end
            // // SIG
            // else begin
            //     if (l2_scale_w_scheme == L2_SCHEME_FP) begin
            //         scaled_l2_psum[sub_b*SCALED_L2_PSUM_WIDTH +: SCALED_L2_PSUM_WIDTH] = ((sub_block_psum_scaled_x[sub_b][L2_PSUM_WIDTH+L2_SCALE_X_SHIFT -: (L2_PSUM_WIDTH+L2_SCALE_X_SHIFT-3)]) * $signed({2'b01, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT], 1'b0}));
            //     end
            //     else if (l2_scale_w_scheme == L2_SCHEME_FP_10) begin
            //         scaled_l2_psum[sub_b*SCALED_L2_PSUM_WIDTH +: SCALED_L2_PSUM_WIDTH] = ((sub_block_psum_scaled_x[sub_b][L2_PSUM_WIDTH+L2_SCALE_X_SHIFT -: (L2_PSUM_WIDTH+L2_SCALE_X_SHIFT-3)]) * $signed({3'b010, l2_scale_w[sub_b*L2_SCALE_W_BIT +: L2_SCALE_W_BIT]}));
            //     end
            // end
        end
    end

    // L2 adder tree
    // Use adder tree with SUB_B input
    generate
        if (B == 128) begin
            if (SUB_B == 64) begin
                ADDER_TREE_2 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 32) begin
                ADDER_TREE_4 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 16) begin
                ADDER_TREE_8 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 8) begin
                ADDER_TREE_16 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 4) begin
                ADDER_TREE_32 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 2) begin
                ADDER_TREE_64 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
        end
        else if (B == 64) begin
            if (SUB_B == 32) begin
                ADDER_TREE_2 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 16) begin
                ADDER_TREE_4 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
            else if (SUB_B == 8) begin
                ADDER_TREE_8 #(.N(NUM_SUB_B), .LEVELS(L2_ADDER_TREE_LEVEL), .WIDTH(SCALED_L2_PSUM_WIDTH)) l2_adder_tree (
                    .fixed_point_in(scaled_l2_psum),
                    .sum(psum)
                );
            end
        end

    endgenerate

    // assign PSUM = psum;

    // ------------------------------------------------------------------------
    // Register output
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        for (int i = 0; i < NUM_SUB_B; i+=1) begin
            sub_block_psum_r[i] <= sub_block_psum[i];
        end
        // <= scaled_l2_psum;
        PSUM <= psum;
    end
endmodule

/*
    FP multipliers and convert to fixed point, WXAY
*/
module FP_mul_vector #(
    parameter N = 4, 
    parameter X = 4, 
    parameter Y = 4,
    parameter PRODUCT_WIDTH = (X-2)+(Y-2)+5
) (
    input  wire [X*N-1:0] WEIGHT,
    input  wire [Y*N-1:0] ACT,
    output wire [PRODUCT_WIDTH*N-1:0] PRODUCT
);

    localparam [1:0] FP4_EXP_BIAS = 2'b01;
    // ------------------------------------------------------------------------
    // Per-element fields (global [0:N-1] arrays)
    // ------------------------------------------------------------------------
    wire                act_s       [0:N-1];
    wire [1:0]          act_e_pre   [0:N-1];
    wire [Y-4:0]        act_m_pre   [0:N-1];
    wire                wgt_s       [0:N-1];
    wire [1:0]          wgt_e_pre   [0:N-1];
    wire [X-4:0]        wgt_m_pre   [0:N-1];

    wire [1:0]          act_e       [0:N-1];
    wire [1:0]          wgt_e       [0:N-1];
    wire [Y-3:0]        act_m       [0:N-1];
    wire [X-3:0]        wgt_m       [0:N-1];

    wire [(X-2)*(Y-2)-1:0]        p_raw       [0:N-1];    // unsigned product
    wire signed [(X-2)*(Y-2):0]   p_sgn       [0:N-1];    // signed product
    wire [2:0]          shamt       [0:N-1];    // 0-4 left-shift amount

    // ------------------------------------------------------------------------
    // Per-element combinational logic (only assign statements inside generate)
    // ------------------------------------------------------------------------
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign {act_s[g], act_e_pre[g], act_m_pre[g]} = ACT   [Y*g +: Y];
            assign {wgt_s[g], wgt_e_pre[g], wgt_m_pre[g]} = WEIGHT[X*g +: X];

            assign act_e[g] = (act_e_pre[g] == 2'b00) ? 2'b00 : act_e_pre[g] - FP4_EXP_BIAS; // FP4 subnormal exp = 0
            assign wgt_e[g] = (wgt_e_pre[g] == 2'b00) ? 2'b00 : wgt_e_pre[g] - FP4_EXP_BIAS; // FP4 subnormal exp = 0

            assign act_m[g] = { (act_e_pre[g] != 2'b00), act_m_pre[g] }; // leading 1 for normal, 0 for subnormal
            assign wgt_m[g] = { (wgt_e_pre[g] != 2'b00), wgt_m_pre[g] }; // leading 1 for normal, 0 for subnormal

            assign p_raw[g]  = act_m[g] * wgt_m[g];
            assign p_sgn[g]  = (act_s[g] ^ wgt_s[g]) ? -$signed({1'b0,p_raw[g]})
                                                     :  $signed({1'b0,p_raw[g]});
            assign shamt[g]  = act_e[g] + wgt_e[g];

            // FP product -> integer product
            assign PRODUCT[g*PRODUCT_WIDTH +: PRODUCT_WIDTH] = (shamt[g] == 3'd0) ? {{4{p_sgn[g][(X-2)*(Y-2)]}}, p_sgn[g]        } :
                                                               (shamt[g] == 3'd1) ? {{3{p_sgn[g][(X-2)*(Y-2)]}}, p_sgn[g], 1'b0  } :
                                                               (shamt[g] == 3'd2) ? {{2{p_sgn[g][(X-2)*(Y-2)]}}, p_sgn[g], 2'b0  } :
                                                               (shamt[g] == 3'd3) ? {{1{p_sgn[g][(X-2)*(Y-2)]}}, p_sgn[g], 3'b0  } :
                                                                                    {                            p_sgn[g], 4'b0  } ;
        end
    endgenerate

endmodule


module ADDER_TREE_2 #(
    parameter N = 2,
    parameter LEVELS = 1,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);
    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-2 : final WIDTH+2 bit sum
    assign sum = lvl0[0] + lvl0[1];
endmodule

module ADDER_TREE_4 #(
    parameter N = 4,
    parameter LEVELS = 2,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);
    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-1 : 4 ? 2 sums (WIDTH+1 bit)
    wire signed [WIDTH:0] lvl1 [0:1];
    assign lvl1[0] = lvl0[0] + lvl0[1];
    assign lvl1[1] = lvl0[2] + lvl0[3];

    // Level-2 : final WIDTH+2 bit sum
    assign sum = lvl1[0] + lvl1[1];
endmodule

module ADDER_TREE_8 #(
    parameter N = 8,
    parameter LEVELS = 3,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);

    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-1 : 8 ? 4 sums (WIDTH+1 bit)
    wire signed [WIDTH:0] lvl1 [0:3];
    generate
        for (g = 0; g < 4; g = g + 1) begin : L1
            assign lvl1[g] = lvl0[2*g] + lvl0[2*g+1];
        end
    endgenerate

    // Level-2 : 4 ? 2 sums (WIDTH+2 bit)
    wire signed [WIDTH+1:0] lvl2 [0:1];
    assign lvl2[0] = lvl1[0] + lvl1[1];
    assign lvl2[1] = lvl1[2] + lvl1[3];

    // Level-3 : final WIDTH+3 bit sum
    assign sum = lvl2[0] + lvl2[1];
endmodule

module ADDER_TREE_16 #(
    parameter N = 16,
    parameter LEVELS = 4,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);

    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-1 : 16 ? 8 sums (WIDTH+1 bit)
    wire signed [WIDTH:0] lvl1 [0:7];
    generate
        for (g = 0; g < 8; g = g + 1) begin : L1
            assign lvl1[g] = lvl0[2*g] + lvl0[2*g+1];
        end
    endgenerate

    // Level-2 : 8 ? 4 sums (WIDTH+2 bit)
    wire signed [WIDTH+1:0] lvl2 [0:3];
    generate
        for (g = 0; g < 4; g = g + 1) begin : L2
            assign lvl2[g] = lvl1[2*g] + lvl1[2*g+1];
        end
    endgenerate

    // Level-3 : 4 ? 2 sums (WIDTH+3 bit)
    wire signed [WIDTH+2:0] lvl3 [0:1];
    assign lvl3[0] = lvl2[0] + lvl2[1];
    assign lvl3[1] = lvl2[2] + lvl2[3];

    // Level-4 : final WIDTH+4 bit sum
    assign sum = lvl3[0] + lvl3[1];
endmodule

module ADDER_TREE_32 #(
    parameter N = 32,
    parameter LEVELS = 5,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);

    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-1 : 32 ? 16 sums (WIDTH+1 bit)
    wire signed [WIDTH:0] lvl1 [0:15];
    generate
        for (g = 0; g < 16; g = g + 1) begin : L1
            assign lvl1[g] = lvl0[2*g] + lvl0[2*g+1];
        end
    endgenerate

    // Level-2 : 16 ? 8 sums (WIDTH+2 bit)
    wire signed [WIDTH+1:0] lvl2 [0:7];
    generate
        for (g = 0; g < 8; g = g + 1) begin : L2
            assign lvl2[g] = lvl1[2*g] + lvl1[2*g+1];
        end
    endgenerate

    // Level-3 : 8 ? 4 sums (WIDTH+3 bit)
    wire signed [WIDTH+2:0] lvl3 [0:3];
    generate
        for (g = 0; g < 4; g = g + 1) begin : L3
            assign lvl3[g] = lvl2[2*g] + lvl2[2*g+1];
        end
    endgenerate

    // Level-4 : 4 ? 2 sums (WIDTH+4 bit)
    wire signed [WIDTH+3:0] lvl4 [0:1];
    assign lvl4[0] = lvl3[0] + lvl3[1];
    assign lvl4[1] = lvl3[2] + lvl3[3];

    // Level-5 : final WIDTH+5 bit sum
    assign sum = lvl4[0] + lvl4[1];
endmodule

module ADDER_TREE_64 #(
    parameter N = 64,
    parameter LEVELS = 6,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);

    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-1 : 64 ? 32 sums (WIDTH+1 bit)
    wire signed [WIDTH:0] lvl1 [0:31];
    generate
        for (g = 0; g < 32; g = g + 1) begin : L1
            assign lvl1[g] = lvl0[2*g] + lvl0[2*g+1];
        end
    endgenerate

    // Level-2 : 32 ? 16 sums (WIDTH+2 bit)
    wire signed [WIDTH+1:0] lvl2 [0:15];
    generate
        for (g = 0; g < 16; g = g + 1) begin : L2
            assign lvl2[g] = lvl1[2*g] + lvl1[2*g+1];
        end
    endgenerate

    // Level-3 : 16 ? 8 sums (WIDTH+3 bit)
    wire signed [WIDTH+2:0] lvl3 [0:7];
    generate
        for (g = 0; g < 8; g = g + 1) begin : L3
            assign lvl3[g] = lvl2[2*g] + lvl2[2*g+1];
        end
    endgenerate

    // Level-4 : 8 ? 4 sums (WIDTH+4 bit)
    wire signed [WIDTH+3:0] lvl4 [0:3];
    generate
        for (g = 0; g < 4; g = g + 1) begin : L4
            assign lvl4[g] = lvl3[2*g] + lvl3[2*g+1];
        end
    endgenerate

    // Level-5 : 4 ? 2 sums (WIDTH+5 bit)
    wire signed [WIDTH+4:0] lvl5 [0:1];
    assign lvl5[0] = lvl4[0] + lvl4[1];
    assign lvl5[1] = lvl4[2] + lvl4[3];

    // Level-6 : final WIDTH+6 bit sum
    assign sum = lvl5[0] + lvl5[1];
endmodule

module ADDER_TREE_128 #(
    parameter N = 128,
    parameter LEVELS = 7,
    parameter WIDTH = 13
    ) (
    input  wire [WIDTH*N-1:0]             fixed_point_in,
    output reg  signed [WIDTH+LEVELS-1:0] sum
);

    // ------------------------------------------------------------------------
    // Arrange input into arrays
    // ------------------------------------------------------------------------
    wire [WIDTH-1:0] in_arr [0:N-1];
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : ELEM
            assign in_arr[g] = fixed_point_in[WIDTH*g +: WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Adder-tree accumulation (each level +1 bit growth)
    // ------------------------------------------------------------------------
    // Level-0 : keep original WIDTH bit terms
    wire signed [WIDTH-1:0] lvl0 [0:N-1];
    generate
        for (g = 0; g < N; g = g + 1) begin : EXT
            assign lvl0[g] = in_arr[g];  // no extra sign-extension here
        end
    endgenerate

    // Level-1 : 128 ? 64 sums (WIDTH+1 bit)
    wire signed [WIDTH:0] lvl1 [0:63];
    generate
        for (g = 0; g < 64; g = g + 1) begin : L1
            assign lvl1[g] = lvl0[2*g] + lvl0[2*g+1];
        end
    endgenerate

    // Level-2 : 64 ? 32 sums (WIDTH+1 bit)
    wire signed [WIDTH+1:0] lvl2 [0:31];
    generate
        for (g = 0; g < 32; g = g + 1) begin : L2
            assign lvl2[g] = lvl1[2*g] + lvl1[2*g+1];
        end
    endgenerate

    // Level-3 : 32 ? 16 sums (WIDTH+2 bit)
    wire signed [WIDTH+2:0] lvl3 [0:15];
    generate
        for (g = 0; g < 16; g = g + 1) begin : L3
            assign lvl3[g] = lvl2[2*g] + lvl2[2*g+1];
        end
    endgenerate

    // Level-4 : 16 ? 8 sums (WIDTH+3 bit)
    wire signed [WIDTH+3:0] lvl4 [0:7];
    generate
        for (g = 0; g < 8; g = g + 1) begin : L4
            assign lvl4[g] = lvl3[2*g] + lvl3[2*g+1];
        end
    endgenerate

    // Level-5 : 8 ? 4 sums (WIDTH+4 bit)
    wire signed [WIDTH+4:0] lvl5 [0:3];
    generate
        for (g = 0; g < 4; g = g + 1) begin : L5
            assign lvl5[g] = lvl4[2*g] + lvl4[2*g+1];
        end
    endgenerate

    // Level-6 : 4 ? 2 sums (WIDTH+6 bit)
    wire signed [WIDTH+5:0] lvl6 [0:1];
    assign lvl6[0] = lvl5[0] + lvl5[1];
    assign lvl6[1] = lvl5[2] + lvl5[3];

    // Level-7 : final WIDTH+7 bit sum
    assign sum = lvl6[0] + lvl6[1];
endmodule

