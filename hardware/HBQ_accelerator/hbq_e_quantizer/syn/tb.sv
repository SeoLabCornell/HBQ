`default_nettype none
`timescale 1ns/1ps

`define CASES 8

module tb;
localparam TIME_OUT_CYCLE = 10000;
localparam CLK_PERIOD = 2;
localparam GROUP_N = `GROUP_NUMBER;

initial begin
  $dumpfile("tb.vcd");
  $dumpvars(0, tb);
end

logic  clk;
logic  rst_n;

reg [`FP_DATAWIDTH-1:0] fp_data [0:`CASES-1][0:GROUP_N-1];
reg [`LP_DATAWIDTH-1:0] lp_data_golden [0:`CASES-1][0:GROUP_N-1];
reg [`SCALE_DATAWIDTH-1:0] lp_scale_golden [0:`CASES-1][0:0];

integer case_idx;
initial begin
    for (case_idx = 0; case_idx < `CASES; case_idx = case_idx + 1) begin
        $readmemb($sformatf("<HOME>/project/nvfp/nv_quantizer/testcase/case%0d/fp.txt", case_idx+1), fp_data[case_idx]);
        $readmemb($sformatf("<HOME>/project/nvfp/nv_quantizer/testcase/case%0d/lp.txt", case_idx+1), lp_data_golden[case_idx]);
        $readmemb($sformatf("<HOME>/project/nvfp/nv_quantizer/testcase/case%0d/scale.txt", case_idx+1), lp_scale_golden[case_idx]);
    end
end

// DUT's input and output
reg [`FP_DATAWIDTH*GROUP_N-1:0] FP_DATA;
wire [`LP_DATAWIDTH*GROUP_N-1:0] LP_DATA;
wire [`SCALE_DATAWIDTH-1:0] LP_SCALE;

FP2GS dut(
  .clk(clk),
  .rst_n(rst_n),
  .FP_DATA(FP_DATA),
  .LP_DATA(LP_DATA),
  .LP_SCALE(LP_SCALE)
);

always #(CLK_PERIOD/2) clk=~clk;
integer error_count = 0;
integer group_idx;

initial begin
  #1 rst_n = 1'bx; clk = 1'bx;

  #(CLK_PERIOD*3) rst_n = 0;
  #(CLK_PERIOD*3) clk = 0; rst_n = 1;
  #(CLK_PERIOD);

  for (case_idx = 0; case_idx < `CASES; case_idx = case_idx + 1) begin
    // Load data for current case
    for (group_idx = 0; group_idx < GROUP_N; group_idx = group_idx + 1) begin
        FP_DATA[`FP_DATAWIDTH*(group_idx+1)-1-:`FP_DATAWIDTH] = fp_data[case_idx][group_idx];
    end
    #(CLK_PERIOD);
    @(negedge clk);
    
    $display("Case %d", case_idx);
    $display("LP Scale output: %d", LP_SCALE);
    $display("LP Scale Golden: %d", lp_scale_golden[case_idx][0]);
    if (LP_SCALE != lp_scale_golden[case_idx][0]) begin
        $display("LP Scale is wrong");
        error_count++;
    end
    for (group_idx = 0; group_idx < GROUP_N; group_idx=group_idx+1) begin
        $display("LP Data[%0d]: %d", group_idx, LP_DATA[`LP_DATAWIDTH*(group_idx+1)-1-:`LP_DATAWIDTH]);
        $display("LP Data[%0d] Golden: %d", group_idx, lp_data_golden[case_idx][group_idx]);
        if (LP_DATA[`LP_DATAWIDTH*(group_idx+1)-1-:`LP_DATAWIDTH] != lp_data_golden[case_idx][group_idx]) begin
            $display("LP Data[%0d] is wrong", group_idx);
            error_count++;
        end
    end
  end

  if (error_count == 0) begin
    $display("--------------------------------");
    $display("Passed all %d test cases :)!!!", `CASES);
    $display("  /\\_/\\");
    $display(" ( ^_^ )");
    $display(" / > < \\");
    $display("--------------------------------");
  end else begin
    $display("--------------------------------");
    $display("%d tests failed QQ", error_count);
    $display(" /\\_/\\");
    $display("( T_T )");
    $display(" > ^ < ");
    $display("--------------------------------");
  end
  #1000; $finish;
end

endmodule
`default_nettype wire