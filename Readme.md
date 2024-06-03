# Sweeping plane GPU computing project
- student : Maxime Pellichero
- 000493288
- maxime.pellichero@ulb.be
- MA2 IRIFS
## How to build
To build a release version open the terminal in the ./build directory and then enter :
```
cmake ../
make
``` 
To build a debug version open the terminal in ./build_debug directory and enter :
```
cmake -DCMAKE_BUILD_TYPE=Debug ../
make
``` 
## How to run
Execute: 
```
<build directory>/bin/PlaneSweep <Sweeping function>
```
The Sweeping plane functions are :
- CPU
- GPU1
- GPU2

GPU3 exists but it doesn' work. See the report for more details.

## Project structure
- The src folder contain all the .hpp and .cpp files
- The kernels folder contain all the .cu and .cuh files
- build contain the normal build
- build_debug contain the debug build