# MSc25-LLM

# 

do the following to enable vitis in the terminal: 
```
source /mnt/ccnas2/bdp/opt/Xilinx/Vitis_HLS/2022.1/settings64.sh
source /mnt/ccnas2/bdp/opt/Xilinx/Vitis/2022.1/settings64.sh
source /mnt/ccnas2/bdp/opt/Xilinx/Vivado/2022.1/settings64.sh
```

To run the simulation for a specific model (e.g., 42M), please follow these steps:
```
cd llama2
cp /mnt/ccnas2/bdp/opt/temp/modelq_42M ./modelq_42M
mv modelq_42M modelq.bin
```
Make sure llama2/firmware/config.h is updated to match the parameters in the corresponding config file under llama2_config.

run the c-simulation: 
```
vitis_hls -f build_prj.tcl csim=1
```
Also, please make sure the corresponding quantized model is placed in the root directory of llama2, and that llama2/firmware/config.h is modified to match the parameters in the corresponding config file under llama2_config.
