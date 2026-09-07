
##################################################################
#    setup, read design and library                              #
##################################################################

set clock_period 2

set search_path [list "." \
    "<PDK_PATH>" \
    "<SYNOPSYS_PATH>/libraries/syn" \
    "<HOME>/project/PROJECT_NAME/syn/output" ]

# set library
set link_path [list "*" \
    "<PDK_PATH>/tcbn28hpcplusbwp30p140ssg0p81v125c.db"]

read_db [list   tcbn28hpcplusbwp30p140ssg0p81v125c.db ]

# set library
set TARGET_LIBS [list "*" \
    "<PDK_PATH>/tcbn28hpcplusbwp30p140ssg0p81v125c.db" ]

set_app_var target_library [concat $TARGET_LIBS]
set_app_var link_library [concat "*" $TARGET_LIBS]
set svr_enable_vpp true
set verilog_files [list "<HOME>/project/PROJECT_NAME/syn/output/pe.$clock_period.28nm.syn.v" 
					]
read_verilog $verilog_files
# read_ddc <HOME>/project/PROJECT_NAME/syn/output/pe.ddc
set top_level "pe"
current_design $top_level
link_design
read_sdc <HOME>/project/PROJECT_NAME/syn/output/pe.${clock_period}.syn.28nm.sdc

# Set top level name

#read_ddc <HOME>/MY_PROJECT/syn_script/pacific_example/output/RP_SIMD.ddc
#read_sdc <HOME>/MY_PROJECT/syn_script/pacific_example/output/RP_SIMD.2.5.syn.28nm.sdc

#read_ddc <HOME>/proj/basic_MAC/syn_8bit/output/top_MAC.2150M.ddc
# read_ddc <HOME>/proj/basic_MAC/syn_8bit/output/top_MAC.${clock_period}M.ddc
#read_sdc <HOME>/proj/basic_MAC/syn_8bit/output/top_MAC.2150M.syn.28nm.sdc




##################################################################
#    Setting Derate and CRPR Section                             #
##################################################################

set timing_remove_clock_reconvergence_pessimism true

##################################################################
#    POWER ANALYSIS	                                         #
##################################################################
set power_enable_analysis true


##################################################################
#    read_sdc and read_parasitics				                         #
##################################################################

#read_sdc
#read_parasitics

##################################################################
#    read_saif				                         #
##################################################################

#read_vcd

#read_saif rtl.saif -rtl_direct -strip_path tb/top_inst 
#read_saif  <HOME>/MY_PROJECT/RP_SIMD/URP_SIMD.saif  -strip_path TB_RP_SIMD_CTRL/URP_SIMD
# read_saif  <HOME>/project/PROJECT_NAME/syn/tb.saif -strip_path tb/dut
read_vcd  <HOME>/project/PROJECT_NAME/syn/tb.vcd -time {20 50} -strip_path tb/dut

#read_saif -input <HOME>/MY_PROJECT/MXDY_training/RP_SIMD.saif -instance TB_PreprocessMX/URP_SIMD

##################################################################
#    Update_timing and check_timing Section                      #
##################################################################

update_power
update_timing -full

# make below as comments for simple 5/31 by MUN
#update_timing -full
#check_timing -verbose > rpt/power/00_check_timing.report

##################################################################
#    Save_Session Section                                        #
##################################################################
save_session pe_syn_power


##################################################################
#    Report_timing Section                                       #
##################################################################
# make below as comments for simple 5/31 by MUN
#report_global_timing > rpt/power/01_report_global_timing
#report_clock -skew -attribute > rpt/power/02_report_clock
#report_analysis_coverage > rpt/power/03_report_analysis_coverage
#report_timing -slack_lesser_than 0.0 -delay min_max -nosplit -input -net  > rpt/power/04_report_timing

# generates average cycle waveforms
# this is available for newer version
#create_power_waveforms

# reports average power results

report_power > ./pe.${clock_period}.report_power

report_annotated_power > ./pe.${clock_period}.report.annotated_power
# below command the same but it is from DC
#report_saif -hierarchy -missing -rtl_saif > ./rpt/power/memory_wrapper_rf1_hd_256x128m2.report.saif

#report_analysis_coverage
#write_sdf memory_wrapper_rf1_hd_256x128m2.sdf_PT
#exit

