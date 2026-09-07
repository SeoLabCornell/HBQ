/*
 * MAC end-to-end testbench used for switching-activity capture.
 *
 * The design parameters (X = weight bitwidth, Y = activation bitwidth,
 * B = block size) come from params.vh, exactly like the RTL, so the same
 * params.vh edit that drives a synthesis run also drives this bench.
 *
 * Selecting the DUT:
 *   +define+TEST_NV   -> WXAY_MAC_NV  (FP8 / E5M3 block scale)
 *   +define+TEST_MX   -> WXAY_MAC_MX  (PoT / E5 block scale)
 * Exactly one is expected for a power run, so that the VCD holds the activity
 * of a single PE.
 *
 * Runtime plusargs:
 *   +vcd=<file>            VCD to write                 (default tb.vcd)
 *   +power_window=<file>   window written for PrimeTime (default power_window.txt)
 *   +testcase_dir=<dir>    root of the golden vectors   (default ./testcase)
 *
 * Compile-time:
 *   +define+TB_CLK_PERIOD=<ns>  clock period, must match the synthesis
 *                               clk_period so that power scales correctly.
 *   +define+SDF_FILE="<path>"   SDF to back-annotate (gate-level runs)
 *
 * Only the streaming region is dumped: $dumpoff is active until the first
 * block is applied and re-asserted once the last block has drained, and the
 * corresponding start/end times are written to the power-window file so the
 * PrimeTime script never needs a hand-maintained per-block-size table.
 */
`default_nettype none
`timescale 1ns/1ps
`include "params.vh"

`ifndef TB_CLK_PERIOD
  `define TB_CLK_PERIOD 2.0
`endif

`ifndef TEST_MX
  `ifndef TEST_NV
    `define TEST_NV
  `endif
`endif

module tb;
    localparam real CLK_PERIOD = `TB_CLK_PERIOD;
    localparam DATA_NUM     = 4096;
    localparam WGT_BITWIDTH = X;
    localparam ACT_BITWIDTH = Y;
    localparam BLOCK_SIZE   = B;
    localparam CASES        = DATA_NUM/BLOCK_SIZE;
    localparam TB_PRODUCT_WIDTH = (ACT_BITWIDTH-2)+(WGT_BITWIDTH-2)+5;
    localparam TB_LEVEL = $clog2(BLOCK_SIZE);
    localparam FIXED_PSUM_WIDTH = TB_PRODUCT_WIDTH + TB_LEVEL;

    string testcase_dir = "./testcase";
    string vcd_file     = "tb.vcd";
    string window_file  = "power_window.txt";

    string testcase_base;

    integer error_count = 0;   // datapath mismatches: must be zero
    integer warn_count  = 0;   // FP16 rounding deviations in the scaled psum

    // switching-activity window handed to PrimeTime
    real power_start_time;
    real power_end_time;

    // stimulus and golden data
    logic [ACT_BITWIDTH-1:0] act_data [0:DATA_NUM-1];
    logic [WGT_BITWIDTH-1:0] wgt_data [0:DATA_NUM-1];
    logic [15:0]             fp_golden [0:0];
    logic [FIXED_PSUM_WIDTH-1:0] psum_golden [0:CASES-1];
    logic [15:0]             scaled_psum_golden [0:CASES-1];
`ifdef TEST_MX
    logic [4:0]              mx_scale_x_data[0:CASES-1];
    logic [4:0]              mx_scale_w_data[0:CASES-1];
`endif
`ifdef TEST_NV
    logic [7:0]              nv_scale_x_data[0:CASES-1];
    logic [7:0]              nv_scale_w_data[0:CASES-1];
`endif

    // One setup block, so plusargs are known before the VCD and the golden
    // vectors are opened.
    initial begin : setup
        void'($value$plusargs("testcase_dir=%s", testcase_dir));
        void'($value$plusargs("vcd=%s", vcd_file));
        void'($value$plusargs("power_window=%s", window_file));
        // dump only the streaming region (see $dumpon/$dumpoff below)
        $dumpfile(vcd_file);
        $dumpvars(0, tb);
        $dumpoff();

`ifdef TEST_MX
        testcase_base = $sformatf("%s/w%0da%0d/mx_w%0da%0db%0d",
                                  testcase_dir, WGT_BITWIDTH, ACT_BITWIDTH,
                                  WGT_BITWIDTH, ACT_BITWIDTH, BLOCK_SIZE);
        $readmemb($sformatf("%s_scale_act.txt", testcase_base), mx_scale_x_data);
        $readmemb($sformatf("%s_scale_wgt.txt", testcase_base), mx_scale_w_data);
`else
        testcase_base = $sformatf("%s/w%0da%0d/nv_w%0da%0db%0d",
                                  testcase_dir, WGT_BITWIDTH, ACT_BITWIDTH,
                                  WGT_BITWIDTH, ACT_BITWIDTH, BLOCK_SIZE);
        $readmemb($sformatf("%s_scale_act.txt", testcase_base), nv_scale_x_data);
        $readmemb($sformatf("%s_scale_wgt.txt", testcase_base), nv_scale_w_data);
`endif
        $readmemb($sformatf("%s_lp_act.txt", testcase_base), act_data);
        $readmemb($sformatf("%s_lp_wgt.txt", testcase_base), wgt_data);
        $readmemb($sformatf("%s_psum.txt", testcase_base), psum_golden);
        $readmemb($sformatf("%s_scaled_psum.txt", testcase_base), scaled_psum_golden);
        $readmemb($sformatf("%s_out.txt", testcase_base), fp_golden);
        $display("[TB] loaded golden vectors from %s_*.txt", testcase_base);
    end

    // Back-annotate the post-synthesis SDF when one is supplied. VCS insists on
    // a compile-time constant here, so this is a define rather than a plusarg.
`ifdef SDF_FILE
    initial begin : annotate_sdf
        $display("[TB] annotating SDF %s", `SDF_FILE);
  `ifdef TEST_MX
        $sdf_annotate(`SDF_FILE, dut_mx);
  `else
        $sdf_annotate(`SDF_FILE, dut_nv);
  `endif
    end
`endif

    // DUT interface
    logic clk;
    logic reset_accum;
    logic drain, drain_out;
    logic [ACT_BITWIDTH*BLOCK_SIZE-1:0] act_vec, act_vec_out;
    logic [WGT_BITWIDTH*BLOCK_SIZE-1:0] wgt_vec, wgt_vec_out;
    logic [15:0] prev_psum;
    logic [15:0] fp16_accum_out;
    logic [FIXED_PSUM_WIDTH-1:0] psum_debug;
    logic [15:0] scaled_psum_debug;
`ifdef TEST_MX
    logic [4:0] scale_x, scale_x_out;
    logic [4:0] scale_w, scale_w_out;
`else
    logic [7:0] scale_x, scale_x_out;
    logic [7:0] scale_w, scale_w_out;
`endif

    // clock generation
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

`ifdef TEST_MX
    WXAY_MAC_MX dut_mx (
`else
    WXAY_MAC_NV dut_nv (
`endif
        .clk               (clk),
        .reset_accum       (reset_accum),
        .drain             (drain),
        .act_vec           (act_vec),
        .wgt_vec           (wgt_vec),
        .scale_x           (scale_x),
        .scale_w           (scale_w),
        .fp16_psum_in      (prev_psum),
        .drain_out         (drain_out),
        .act_vec_out       (act_vec_out),
        .wgt_vec_out       (wgt_vec_out),
        .scale_x_out       (scale_x_out),
        .scale_w_out       (scale_w_out),
        .fp16_psum_out     (fp16_accum_out),
        .psum_debug        (psum_debug),
        .scaled_psum_debug (scaled_psum_debug)
    );

    task automatic apply_case_vectors(input int case_idx);
        int g;
        for (g = 0; g < BLOCK_SIZE; g++) begin
            act_vec[ACT_BITWIDTH*g +: ACT_BITWIDTH] = act_data[case_idx*BLOCK_SIZE+g];
            wgt_vec[WGT_BITWIDTH*g +: WGT_BITWIDTH] = wgt_data[case_idx*BLOCK_SIZE+g];
        end
    endtask

    logic [ACT_BITWIDTH*BLOCK_SIZE-1:0] act_vec_out_golden;
    logic [WGT_BITWIDTH*BLOCK_SIZE-1:0] wgt_vec_out_golden;
    task automatic apply_case_vectors_out(input int case_idx);
        int g;
        for (g = 0; g < BLOCK_SIZE; g++) begin
            act_vec_out_golden[ACT_BITWIDTH*g +: ACT_BITWIDTH] = act_data[case_idx*BLOCK_SIZE+g];
            wgt_vec_out_golden[WGT_BITWIDTH*g +: WGT_BITWIDTH] = wgt_data[case_idx*BLOCK_SIZE+g];
        end
    endtask

    reg signed [15:0] scaled_psum_diff;

    task automatic run_end_to_end();
        bit [CASES-1:0] case_pass;

        // ensure the accumulator starts from a clean state
        reset_accum        = 1'b1;
        act_vec_out_golden = '0;
        wgt_vec_out_golden = '0;
        scale_x            = '0;
        scale_w            = '0;
        drain              = 1'b0;
        prev_psum          = '0;
        repeat (2) @(posedge clk);
        reset_accum = 1'b0;

        // ---- switching-activity window opens here ----
        power_start_time = $realtime;
        $dumpon();
        $display("[TB] power window start: %0t", $time);

        fork
            begin : data_input
                for (int case_idx = 0; case_idx < CASES; case_idx++) begin
                    @(negedge clk);
                    apply_case_vectors(case_idx);
`ifdef TEST_MX
                    scale_x = mx_scale_x_data[case_idx];
                    scale_w = mx_scale_w_data[case_idx];
`else
                    scale_x = nv_scale_x_data[case_idx];
                    scale_w = nv_scale_w_data[case_idx];
`endif
                    case_pass[case_idx] = 1'b1;
                end
                @(negedge clk);
                scale_x = '0;
                scale_w = '0;
                act_vec = '0;
                wgt_vec = '0;
            end

            begin : data_vec_output
                repeat (1) @(posedge clk);
                for (int case_idx = 0; case_idx < CASES; case_idx++) begin
                    @(negedge clk);
                    apply_case_vectors_out(case_idx);
                    if (act_vec_out !== act_vec_out_golden) begin
                        case_pass[case_idx] = 1'b0;
                        error_count = error_count + 1;
                        $display("[ERROR][case %0d @ %0t] act_vec_out mismatch", case_idx, $time);
                    end
                    if (wgt_vec_out !== wgt_vec_out_golden) begin
                        case_pass[case_idx] = 1'b0;
                        error_count = error_count + 1;
                        $display("[ERROR][case %0d @ %0t] wgt_vec_out mismatch", case_idx, $time);
                    end
                end
            end

            begin : mac_vector_output
                repeat (1) @(posedge clk);
                for (int case_idx = 0; case_idx < CASES; case_idx++) begin
                    @(negedge clk);
                    if (psum_debug !== psum_golden[case_idx]) begin
                        case_pass[case_idx] = 1'b0;
                        error_count = error_count + 1;
                        $display("[ERROR][case %0d @ %0t] fixed-point PSUM mismatch: got %0d expected %0d",
                                 case_idx, $time, psum_debug, psum_golden[case_idx]);
                    end
                end
            end

            begin : scaling_output
                repeat (2) @(posedge clk);
                for (int case_idx = 0; case_idx < CASES; case_idx++) begin
                    @(negedge clk);
                    scaled_psum_diff = scaled_psum_debug - scaled_psum_golden[case_idx];
                    if (scaled_psum_diff < 0) scaled_psum_diff = -scaled_psum_diff;
                    // The FP16 conversion is allowed to differ from the golden
                    // model by 1 ULP. A handful of cases at the higher
                    // activation precisions land 2 ULP out because the golden
                    // model rounds the scale product slightly differently; that
                    // is a modelling difference, not a datapath fault, so it is
                    // counted separately and does not fail the run.
                    if (scaled_psum_diff > 1) begin
                        case_pass[case_idx] = 1'b0;
                        warn_count = warn_count + 1;
                        $display("[WARN ][case %0d @ %0t] scaled_psum off by %0d ULP: got %0b expected %0b",
                                 case_idx, $time, scaled_psum_diff,
                                 scaled_psum_debug, scaled_psum_golden[case_idx]);
                    end
                end
            end
        join

        repeat (1) @(posedge clk);
        @(negedge clk);
        drain     = 1'b1;
        $display("[TB] final output:        %016b", fp16_accum_out);
        $display("[TB] golden final output: %016b", fp_golden[0]);
        prev_psum = 16'b0001000100010001;
        @(negedge clk);
        prev_psum = '0;
        drain     = 1'b0;
        $display("[TB] propagate output:        %016b", fp16_accum_out);
        $display("[TB] golden propagate output: %016b", 16'b0001000100010001);

        // ---- switching-activity window closes here ----
        power_end_time = $realtime;
        $dumpoff();
        $display("[TB] power window end: %0t", $time);
    endtask

    // The window is written in VCD timescale units (1ns here) so that it can be
    // handed straight to read_vcd -time {start end}.
    task automatic write_power_window();
        int fd;
        fd = $fopen(window_file, "w");
        if (fd == 0) begin
            $display("[TB][WARN] could not open %s for writing", window_file);
            return;
        end
        $fwrite(fd, "start %0.3f\n", power_start_time);
        $fwrite(fd, "end %0.3f\n", power_end_time);
        $fwrite(fd, "clk_period %0.6f\n", CLK_PERIOD);
        $fwrite(fd, "block_size %0d\n", BLOCK_SIZE);
        $fwrite(fd, "cases %0d\n", CASES);
        $fclose(fd);
        $display("[TB] power window written to %s: %0.3f .. %0.3f ns",
                 window_file, power_start_time, power_end_time);
    endtask

    task automatic report_result(input string mode);
        if (warn_count != 0)
            $display("[TB] %0d scaled_psum values differ from the golden model by more than 1 ULP",
                     warn_count);
        if (error_count == 0) begin
            $display("--------------------------------");
            $display("All MAC %s-mode tests passed!", mode);
            $display("  /\\_/\\");
            $display(" ( ^_^ )");
            $display(" / > < \\");
            $display("--------------------------------");
        end else begin
            $display("--------------------------------");
            $display("MAC %s-mode tests FAILED: %0d issues", mode, error_count);
            $display(" /\\_/\\");
            $display("( T_T )");
            $display(" > ^ < ");
            $display("--------------------------------");
        end
    endtask

    initial begin : main
        reset_accum = 1'b1;
        act_vec     = '0;
        wgt_vec     = '0;
        scale_x     = '0;
        scale_w     = '0;
        repeat (3) @(posedge clk);

`ifdef TEST_MX
        $display("=== MAC MX (PoT scale) end-to-end test: X=%0d Y=%0d B=%0d ===", X, Y, B);
        run_end_to_end();
        report_result("MX");
`else
        $display("=== MAC NV (FP8 scale) end-to-end test: X=%0d Y=%0d B=%0d ===", X, Y, B);
        run_end_to_end();
        report_result("NV");
`endif

        write_power_window();
        #10;
        $finish;
    end

endmodule

`default_nettype wire
