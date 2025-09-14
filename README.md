Project setup (Windows + VS2022)
1.Enable CUDA in your project
	Right-click your project → Build Customizations… → tick CUDA 12.9 → OK.
	Make sure Platform = x64 (not Win32).
2. Add CUDA files
	Add new files:
	md_cuda.cu (kernels + device code)
	md_cuda_iface.h (host-visible API)
	md_cuda_iface.cpp (thin wrappers calling CUDA)
	VS auto-detects .cu as CUDA C/C++ if build customization is enabled.
3. Project Properties (Release & Debug)
	C/C++ → Language → C++ Language Standard: ISO C++17
	C/C++ → Code Generation → Runtime Library: /MD (Multi-threaded DLL)
	CUDA C/C++ → Device → Code Generation: compute_86,sm_86 (for rtx3050)
	CUDA C/C++ → Common → Generate Relocatable Device Code: Yes (-rdc=true)
	CUDA Linker → General → Perform Device Link: Yes
	C/C++ → Preprocessor → Preprocessor Definitions: add USE_CUDA (so you can #ifdef GPU paths)
	(Optional later) CUDA C/C++ → Common → Use Fast Math: No (start accurate; turn Yes after validation)
4. Sanity test (tiny ping)
	In md_cuda.cu:
	 #include <cuda_runtime.h>
	__global__ void ping() {}
	extern "C" bool md_cuda_ping() { ping<<<1,1>>>(); return cudaPeekAtLastError()==cudaSuccess; }
	In md_cuda_iface.h:
	 extern "C" bool md_cuda_ping();
	Call md_cuda_ping() from main() (or your class). Build & run → should return true.

1. Double click on the sln file (DSPC/SimpleMD/SimpleMolecularDynamics.sln)
2. Microsoft VS will be opened.
3. Ensure that you already tick the CUDA 12.9 in Build Dependencies -> Build Customization

Repository Directory:
DSPC/``
|-SimpleMD/
| |-SimpleMolecularDynamics/
| | |-AndersonThermostat.cc
| | |-AndersonThermostat.h
| | |-bench_md.cpp 
| | |-example.cc
| | |-example_2d.cc
| | |-mat.cc
| | |-mat.h
| | |-md_cuda.cu
| | |-md_cuda_iface.cpp
| | |-md_cuda_iface.h
| | |-MDParticle.h
| | |-MDParticleList.h
| | |-MDParticleListEntry.h
| | |-MDSim.cc
| | |-MDSim.h
| | |-vec.h
| | |-vec.cc
| |-x64/
| | |-Debug/SimpleMolecularDynamics.exe
| | |-Release-CPU/SimpleMolecularDynamics.exe
| | |-Release-GPU/SimpleMolecularDynamics.exe
|-README.md

NOTES:
0. Only 1 int(main) can be included in the build project.
   Exclude others if you want to successfully build the project. 
1. bench_md.cpp (main) is only for headless benchmarking. It wont 
   display the MD simulation output like the original code do. It 
   will just output the time and pot for Release-CPU and Release-CPU
2. md_cuda.cu is the GPU backend, goal is to build Verlet neighbor
   lists and compute LJ forces on the GPU (double precision), one
   thread per atom, no atomics. 
3. example_2d.cc (main) is the code for displaying and executing 
   the MD simulation. 
4. In MDSim.cc, there are 2 hotspot that needed to be converted to
   CUDA code. The 2 hotspot is considered computationally expensive
   due to nested for loop.

To run the headless benchmark:
1. ctrl + ` to open terminal
2. 
.\x64\Release-CPU\SimpleMolecularDynamics.exe 1024 200 
3. 
.\x64\Release-GPU\SimpleMolecularDynamics.exe 1024 200 


In Visual Studio:

1. Right-click your project → Configuration Manager… → Active solution configuration → <New…>

	Name: Release-GPU-FP32

	Copy settings from: Release-GPU

2. With Release-GPU-FP32 selected:

	C/C++ → Preprocessor → Preprocessor Definitions → add: MD_MIXED_FP32

	CUDA C/C++ → Preprocessor → Preprocessor Definitions → add: MD_MIXED_FP32

3. Build all three:

Release-CPU, Release-GPU, Release-GPU-FP32
