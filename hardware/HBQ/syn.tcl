# Synthesis script for HBQ hardware synthesis result reproduction
# corner: We use tt corner with 0.9V and 25C for synthesis
set_host_options -max_cores 16
set auto_write_pvl false
set auto_write_mr false
set auto_write_syn false


# Load common variables, artisan standard cells
# source -verbose "./script/common.lp.syn.tcl"
# TODO: set your search path here
set search_path [list "." \
                     ]

# set library
# TODO: set your .db file here
set TARGET_LIBS [list \
  ]

set_app_var target_library [concat $TARGET_LIBS]
set_app_var link_library [concat "*" $TARGET_LIBS]
# set link_library "* tcbn28hpcplusbwp30p140ssg0p9vm40c.db"
# set target_library "tcbn28hpcplusbwp30p140ssg0p9vm40c.db"

# Set top level name
set top_level "WXAY_MAC_NV"

# set don't use cells
source dont_use.syn.tcl

# Read verilog files
set VERILOG_DIR "."

set RTL_SRC_FILES [list \
"$VERILOG_DIR/MAC.sv" \
"$VERILOG_DIR/FP16_ADDER.v" \
"$VERILOG_DIR/LP_vector.sv" \
"$VERILOG_DIR/scale.v" \
"$VERILOG_DIR/params.vh"
]

analyze -format sverilog $RTL_SRC_FILES

elaborate $top_level
list_designs
current_design $top_level

link

set clk_period 2
set rpt_file "./log/${top_level}.${clk_period}.28nm.syn"

# clock gating
identify_clock_gating
report_clock_gating -multi_stage -nosplit > ${rpt_file}_clock_gating.28nm.rpt
set_clock_gating_style -positive_edge_logic {integrated} -negative_edge_logic {integrated} -max_fanout 60 

# Clock period
set clk_name "clk"
set clk_port "clk"
create_clock -name clk  -period $clk_period   [get_ports  clk]
#in ns

set clk_uncertainty 0.01
set clk_transition 0.1

set_drive 0 [get_clocks $clk_name]

set_clock_uncertainty $clk_uncertainty [get_clocks $clk_name]
#Propagated clock used for gated clocks only
#set_propagated_clock [get_clocks $clk_name]
set_clock_transition $clk_transition [get_clocks $clk_name]

set_operating_conditions "tt0p9v25c" -library "tcbn28hpcplusbwp30p140tt0p9v25c" 
#set_wire_load_model -name "ibm13_wl10" -library "typical" 
set_wire_load_mode "segmented" 

# set to 10%, 50%
set typical_input_delay_min 0.2
set typical_input_delay_max 1
set typical_output_delay 0.1
set typical_wire_load 0.010 

# Link the design
# link

# Set maximum fanout of gates
set_max_fanout 16 $top_level 

# Configure the clock network
set_fix_hold [all_clocks] 
set_dont_touch_network $clk_port 

set_driving_cell -lib_cell INVD18BWP30P140 [all_inputs]
set_input_delay -max $typical_input_delay_min [all_inputs] -clock $clk_name 
set_input_delay -min $typical_input_delay_max [all_inputs] -clock $clk_name 

remove_input_delay -clock $clk_name [find port $clk_port]
set_output_delay $typical_output_delay [all_outputs] -clock $clk_name 

# Set loading of outputs 
set_load $typical_wire_load [all_outputs] 

# Verify the design
check_design

# set compile_ultra_ungroup_small_hierarchies false
# compile_ultra -no_autoungroup -no_boundary_optimization
compile_ultra -gate_clock -no_autoungroup
# balance_registers

# Rename modules, signals according to the naming rules Used for tool exchange
source "./naming_rules.syn.tcl"
exec mkdir ./output
exec mkdir ./log
# Generate structural verilog netlist
write_file -hierarchy -format verilog -output "./output/${top_level}.${clk_period}.28nm.syn.v"
# Save current design
write_file -hierarchy -format ddc -output "./output/${top_level}.ddc"

# Generate Standard Delay Format (SDF) file
write_sdf -context verilog "./output/${top_level}.${clk_period}.syn.28nm.sdf"

# Generate timing constraints file
write_sdc "./output/${top_level}.${clk_period}.syn.28nm.sdc"

# Generate report file
set maxpaths 100
#set minpaths 100


check_design > ${rpt_file}_chk_design.28nm.rpt
report_area -hierarchy -nosplit > ${rpt_file}_area.28nm.rpt
report_power -hier -analysis_effort medium -nosplit > ${rpt_file}_power.28nm.rpt
report_design -nosplit > ${rpt_file}
report_cell -nosplit > ${rpt_file}
report_port -nosplit -verbose > ${rpt_file}
report_compile_options -nosplit > ${rpt_file}
report_clock_gating -multi_stage -nosplit > ${rpt_file}_auto_clock_gating.28nm.rpt
report_constraint -all_violators -verbose -nosplit > ${rpt_file}
report_timing -path full -delay max -max_paths $maxpaths -nworst 100 -nosplit > ${rpt_file}_timing.28nm.rpt

# Exit dc_shell
quit
