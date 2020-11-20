# Description of full_dataset.csv
The file contains the full raw dataset from which the scenario datasets were created. The column describe the following:

| Column  |  Description  | 
|---|---|
| cpu_name  | name of the used CPU |
| gpu_name  | name of the used GPU |
| fps  | fpsbenchmark: fps value for measurements; userbenchmark: aggregated number from userbenchmark for the measurements |
| sample_count  |  userbenchmark: count of raw measurements (all that are included in the respective histogram); fpsbenchmark: not used |
| fps_sample  |  userbenchmark: raw measurements (all that are included in the respective histogram); fpsbenchmark: not used  |
| dataset  | userbenchmark = 0; fpsbenchmark = 1|
| cpu_* | technical features of the CPU|
| gpu_* | technical features of the GPU|
| game_*  | one-hot encoding of game name|
| setting_* | one-hot encoding of used quality setting for game |
| resolution | displayed resolution of game|

## Description of the technical features
### CPU

| Column  |  Description  | 
|---|---|
| cpu_# of Cores  | number of physical cores |
| cpu_# of Threads  | number of threads|
| cpu_Base Clock  | base clock in Mhz|
| cpu_Cache L1  | size of level 1 cache in kB |
| cpu_Cache L2  |  size of level 2 cache in kB |
| cpu_Cache L3  |  size of level 3 cache in MB|
| cpu_Die Size | physical size of the die in square meter|
| cpu_Frequency | frequency in Mhz|
| cpu_Multiplier | multiplier of CPU|
| cpu_Multiplier Unlocked  | 0=multiplier locked, 1=multiplier unlocked |
| cpu_Process Size |  used process size in meter |
| cpu_SMP # CPUs | number of symmetric multiprocessors|
| cpu_TDP | thermal design power in watt|
| cpu_Transistors | count of transistors in million|
| cpu_Turbo Clock | turbo clock in Mhz|

### GPU

| Column  |  Description  | 
|---|---|
| gpu_Bandwidth  | bandwidth of the GPU in MB/s |
| gpu_Base Clock  | base clock in MHz|
| gpu_Boost Clock | boost clock in MHz |
| gpu_Compute Units | number of computing units|
| gpu_Die Size  |  physical size of die in square meter |
| gpu_Execution Units |  number of execution units |
| gpu_FP32 (float) performance| theoretical Float 32 performance in MFLOP/s |
| gpu_Memory Bus| width of memory bus in bits|
| gpu_Memory Size | size of memory in MB |
| gpu_Pixel Rate | theoretical pixel rate in MPixel/s |
| gpu_Process Size|  used process size in meter |
| gpu_ROPs | number of render output units|
| gpu_Shading Units | number of shading units|
| gpu_TMUs | number of texture mapping units |
| gpu_Texture Rate | theoretical texture rate in KTexel/s|
| gpu_Transistors | number of transistors in million| 
| gpu_Architecture_* | one-hot encoded architecture|
| gpu_Memory Type_*| one-hot encoded memory type|
| gpu_Bus Interface_* | one-hot encoded bus interface|
| gpu_OpenCL_* | one-hot encoded version of OpenCL|
| gpu_Shader Model_* | one-hot encoded version of shader model|
| gpu_Vulkan_* | one-hot encoded version of Vulkan |
| gpu_OpenGL_* | one-hot encoded version of OpenGL|