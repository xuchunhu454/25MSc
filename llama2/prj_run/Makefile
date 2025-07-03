

set_110m: 
	cp /mnt/ccnas2/bdp/opt/temp/modelq_110M.bin ./modelq.bin 
	cp ../../llama2_config/config110M.h ../firmware/config.h 

set_42m: 
	cp /mnt/ccnas2/bdp/opt/temp/modelq_42M.bin ./modelq.bin 
	cp ../../llama2_config/config42M.h ../firmware/config.h 

set_15m: 
	cp /mnt/ccnas2/bdp/opt/temp/modelq_15M.bin ./modelq.bin 
	cp ../../llama2_config/config15M.h ../firmware/config.h 

.PHONY: cpu
cpu: 
	g++ -O3 -o run ../firmware/* ../testbench/* -lm
	./run 

csim:
	vitis_hls -f build_prj.tcl csim=1 

csynth:
	vitis_hls -f build_prj.tcl synth=1 

vsynth:
	vitis_hls -f build_prj.tcl vsynth=1 

synth:
	vitis_hls -f build_prj.tcl synth=1 vsynth=1

.PHONY: clean
clean:
	rm -f run
