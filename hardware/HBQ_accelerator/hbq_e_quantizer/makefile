# modify the following for your design
TOP_DESIGN          = FP2GS
CLOCK_PERIOD        = 2.0  # in the unit of ns
CLOCK_PORT          = clk
RESET_PORT          = rst
SIM_TIMESCALE       = 1ns/1ps
TOP_DIR 			= <HOME>/project/nvfp/hbq_e_quantizer
HEADERS             = $(TOP_DIR)/top.sv # It's recommended to  put all the macro/interface/parameter definition in header file
MEMORY_HEADERS      = $(wildcard)
TESTBENCH           = $(TOP_DIR)/tb.sv
# TESTBENCH     = $(wildcard tb/verilog/tb_top.sv tb/verilog/chip_top.sv )
SYN_FILE            = $(TOP_DIR)/syn/output/*.v*
IO_MODEL            = $(wildcard tb/verilog/SP018V3EBCD*.v)
#NETLIST                = <HOME>/hls/catapult/retnet/syn_script/pacific_example/output/MXDY_CORE.2.5.28nm.syn.v
NETLIST             = $(TOP_DIR)/syn/output/FP2GS.2.28nm.syn.v
# POST_NETLIST        = $(wildcard <HOME>/hls/catapult/retnet/datapath/apr/output/Datapath_rtl.2.apr.v)
SDC_FILE            = ./syn/data/$(TOP_DESIGN).sdc
SIM_FILE            = $(TESTBENCH)
# SIM_FILE          = $(wildcard rtl/*.sv) 
SYNTH_DIR           = $(TOP_DIR)/syn
# export file
export HEADERS
export SYN_FILE
export TOP_DESIGN
export CLOCK_PERIOD
export CLOCK_PORT
export RESET_PORT
export NETLIST
export SDC_FILE
# VCS = vcs -V -sverilog +v2k -full64 -debug_acc+all -kdb -lca -cm line+tgl+assert+branch+cond+fsm -debug_region+cell+encrypt \
#             +sdfverbose -sdfretain -negdelay +neg_tchk +mindelays -R -gui -l  com.log 
VCS = vcs -V -sverilog +v2k -full64 -debug_acc+all -kdb -lca -cm line+tgl+assert+branch+cond+fsm -debug_region+cell+encrypt \
						+sdfverbose -sdfretain -negdelay +neg_tchk +mindelays -l com.log -R -gui
VCS_POST = vcs -V -sverilog -full64 -debug_access+all +lint=TFIPC-L -kdb -lca -cm line+tgl+assert+branch+cond+fsm \
            -sdfretain -negdelay \
            +sdfverbose \
            +neg_tchk \
            +mindelays 
# -P ${NOVAS_HOME}/share/PLI/VCS/LINUX64/novas.tab ${NOVAS_HOME}/share/PLI/VCS/LINUX64/pli.a
LIB = <PDK_PATH>/tcbn28hpcplusbwp30p140.v
# Simulation
.PHONY: all simv verdi
all: simv
#	./simv | tee program.out
simv: $(HEADERS) $(TESTBENCH) 
	$(VCS) $^ +notimingcheck -override_timescale=$(SIM_TIMESCALE) -o simv 
verdi:  
	echo "$(HEADERS) $(SIM_FILE) $(TESTBENCH)" > test.f && \
	verdi -full64 -ssf test.fsdb -F ./test.f &
# Synthesis 
.PHONY: link syn syn_simv verdi_syn 
link:   $(HEADERS) $(SYN_FILE) 
	cd $(SYNTH_DIR) && dc_shell-t -f ./link.tcl | tee syn.out
syn:    $(HEADERS)
	dc_shell-t -f ./syn.tcl | tee syn.out
syn_simv:    $(NETLIST) $(TESTBENCH)
	$(VCS) $^ +notimingcheck+ $(LIB) -diag timescale -override_timescale=$(SIM_TIMESCALE) | tee syn_sim.log
    
verdi_syn:  syn_simv
	echo " $(NETLIST) $(TESTBENCH)" > test.f && \
        verdi -full64 -ssf ./tb/test.fsdb +v2k -sverilog -F ./test.f &
# postsim 
post_simv:  $(POST_NETLIST)
	$(VCS_POST)  $^ $(STD_LIB) +define+case2 +define+sdf_post_wst +sdfverbose -diag timescale -override_timescale=$(SIM_TIMESCALE) -o post_simv | tee post.log
verdi_post:
	echo " $(POST_NETLIST) " > test_post.f && \
        verdi -full64 -ssf ./tb/test_post.fsdb +v2k -sverilog -F ./test_post.f &
.PHONY: post_simv verdi_post
# STA/PT
pt:     $(NETLIST)  $(SDC_FILE)
	cd $(SYNTH_DIR) && pt_shell -f ./pt.tcl | tee pt.out
clean:
	rm -rf *.log csrc simv* *.key *.vpd DVEfile *~ *.fsdb* novas* *Log 
	rm -rf *simv program.out csrc *simv.daidir
	rm -rf syn/*.rpt syn/alib* syn/*.svf syn/*.log syn/*.mr syn/*.out
# rm -rf *.vcd
    
nuke:   clean
	rm -rf syn/data/* syn/reports/* syn/work/*