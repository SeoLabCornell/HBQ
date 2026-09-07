# PrimeTime PX averaged-power analysis for one MAC configuration.
#
# Everything that changes between configurations is passed in through the
# environment, so this script has no paths baked into it and can be re-run for
# every (activation precision, block size, scaling scheme) point of the sweep.
#
#   PT_TOP            top-level module (WXAY_MAC_NV or WXAY_MAC_MX)
#   PT_NETLIST        gate-level netlist written by Design Compiler
#   PT_SDC            constraints written by Design Compiler
#   PT_VCD            VCD captured from the gate-level simulation
#   PT_STRIP_PATH     VCD scope holding the DUT (e.g. tb/dut_nv)
#   PT_VCD_START      start of the switching-activity window, in ns
#   PT_VCD_END        end of the switching-activity window, in ns
#   PT_REPORT_PREFIX  prefix for the two generated reports
#   PT_PDK_DIR        directory holding the TSMC 28nm .db files
#   PT_LIB_DB         .db file to characterize against
#
# run_pt_sweep.py fills all of these in; PT_VCD_START/PT_VCD_END come from the
# power_window.txt that the testbench writes, so the measured window always
# matches the streaming region of the simulation.

proc env_or_die {name} {
    global env
    if {![info exists env($name)]} {
        puts "ERROR: environment variable $name is not set"
        exit 1
    }
    return $env($name)
}

proc env_or {name default} {
    global env
    if {[info exists env($name)]} { return $env($name) }
    return $default
}

set top_level      [env_or_die PT_TOP]
set netlist_file   [env_or_die PT_NETLIST]
set sdc_file       [env_or_die PT_SDC]
set vcd_file       [env_or_die PT_VCD]
set strip_path     [env_or_die PT_STRIP_PATH]
set vcd_start      [env_or_die PT_VCD_START]
set vcd_end        [env_or_die PT_VCD_END]
set report_prefix  [env_or_die PT_REPORT_PREFIX]

# add your timing_power_noise model below
set pdk_dir [env_or PT_PDK_DIR ]

set lib_db  [env_or PT_LIB_DB "tcbn28hpcplusbwp30p140ssg0p81v125c.db"]

##################################################################
#    setup, read design and library                              #
##################################################################

set search_path [list "." $pdk_dir]

set link_path    [list "*" [file join $pdk_dir $lib_db]]
set TARGET_LIBS  [list "*" [file join $pdk_dir $lib_db]]

read_db [list $lib_db]

set_app_var target_library [concat $TARGET_LIBS]
set_app_var link_library   [concat "*" $TARGET_LIBS]
set svr_enable_vpp true

read_verilog [list $netlist_file]
current_design $top_level
link_design
read_sdc $sdc_file

##################################################################
#    Setting Derate and CRPR Section                             #
##################################################################

set timing_remove_clock_reconvergence_pessimism true

##################################################################
#    POWER ANALYSIS                                              #
##################################################################

set power_enable_analysis true

# Switching activity from the gate-level simulation, restricted to the window
# in which blocks are actually being streamed through the PE.
read_vcd $vcd_file -time [list $vcd_start $vcd_end] -strip_path $strip_path

update_power
update_timing -full

##################################################################
#    Report                                                      #
##################################################################

report_power             > ${report_prefix}.report_power
report_power -hierarchy  > ${report_prefix}.report_power_hierarchy

exit
