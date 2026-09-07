/*
    Systolic array connections
*/


`include "params.vh";

module SA (
    input                     clk,
    input                     reset_accum,
    input  [2:0]              mode,
    input                     propagate_wgt, // preload weight
    input                     in_val,
    input  [Y*B-1:0]          act_vec,
    input  [X*B-1:0]          wgt_vec,
    input  [7:0]              scale_x,
    input  [7:0]              scale_w,
    input  [L2_SCALE_X_BIT*NUM_SUB_B-1:0] l2_scale_x,
    input  [L2_SCALE_W_BIT*NUM_SUB_B-1:0] l2_scale_w,
    input  l2_scale_w_scheme,
    
    output reg [16*SA_N-1:0] scaled_psum_out
);

logic [X*B-1:0] sa_wgt_vec[0:SA_N];
logic [7:0]   sa_scale_w[0:SA_N];
logic [L2_SCALE_W_BIT*NUM_SUB_B-1:0] sa_l2_scale_w [0:SA_N];
logic sa_l2_scale_w_scheme [0:SA_N];
logic [15:0] sa_scaled_psum[0:SA_N-1];
logic sa_propagate_wgt[0:SA_N];

// first PE for weight propagation
always_comb begin
    sa_propagate_wgt[0] = propagate_wgt;
    sa_wgt_vec[0] = wgt_vec;
    sa_scale_w[0] = scale_w;
    sa_l2_scale_w[0] = l2_scale_w;
    sa_l2_scale_w_scheme[0] = l2_scale_w_scheme;
end

// SA
genvar w, n;
generate
    for (n = 0; n < SA_N; n=n+1) begin : pe_array // rows
        WXAY_MAC_NV pe (
            // input
            .clk(clk),
            .reset_accum(reset_accum),
            .mode(mode),
            .propagate_wgt(sa_propagate_wgt[n]),
            .act_vec(in_val ? act_vec : '0),
            .wgt_vec(sa_wgt_vec[n]),
            .scale_x(in_val ? scale_x : '0),
            .scale_w(sa_scale_w[n]),
            .l2_scale_x(in_val ? l2_scale_x : '0),
            .l2_scale_w(sa_l2_scale_w[n]),
            .l2_scale_w_scheme(sa_l2_scale_w_scheme[n]),

            // output
            .propagate_wgt_out(sa_propagate_wgt[n+1]),
            .wgt_vec_out(sa_wgt_vec[n+1]),
            .scale_w_out(sa_scale_w[n+1]),
            .l2_scale_w_out(sa_l2_scale_w[n+1]),
            .l2_scale_w_scheme_out(sa_l2_scale_w_scheme[n+1]),
            .scaled_psum_out(sa_scaled_psum[n])
        );
    end
endgenerate


// output, drain vertically
always_comb begin
    for (int i = 0; i < SA_N; i+=1) begin
        scaled_psum_out[i*16 +: 16] = sa_scaled_psum[i];
    end
end

endmodule