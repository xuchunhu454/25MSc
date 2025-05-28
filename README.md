# MSc25-LLM

# 

do the following to enable vitis in the terminal: 
```
source /mnt/ccnas2/bdp/opt/Xilinx/Vitis_HLS/2022.1/settings64.sh
source /mnt/ccnas2/bdp/opt/Xilinx/Vitis/2022.1/settings64.sh
source /mnt/ccnas2/bdp/opt/Xilinx/Vivado/2022.1/settings64.sh
```

then run the following to get the model of llama2 110M. 
```
cd llama2
cp /mnt/ccnas2/bdp/opt/temp/modelq.bin ./
```

run the c-simulation: 
```
vitis_hls -f build_prj.tcl csim=1
```
Also, please make sure the corresponding quantized model is placed in the root directory of llama2, and that llama2/firmware/config.h is modified to match the parameters in the corresponding config file under llama2_config.
