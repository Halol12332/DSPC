# DSPC
1. double click .sln file
2. ctrl + shift + b
3. ctrl + `
4. ./SimpleMolecularDynamics 1000 0.001 1 1 


Project setup (Windows + VS2022)
Enable CUDA in your project
Right-click your project → Build Customizations… → tick CUDA 12.9 → OK.
Make sure Platform = x64 (not Win32).
Add CUDA files
Add new files:
md_cuda.cu (kernels + device code)
md_cuda_iface.h (host-visible API)
md_cuda_iface.cpp (thin wrappers calling CUDA)
VS auto-detects .cu as CUDA C/C++ if build customization is enabled.
Project Properties (Release & Debug)
C/C++ → Language → C++ Language Standard: ISO C++17
C/C++ → Code Generation → Runtime Library: /MD (Multi-threaded DLL)
CUDA C/C++ → Device → Code Generation: compute_86,sm_86 (for rtx3050)
CUDA C/C++ → Common → Generate Relocatable Device Code: Yes (-rdc=true)
CUDA Linker → General → Perform Device Link: Yes
C/C++ → Preprocessor → Preprocessor Definitions: add USE_CUDA (so you can #ifdef GPU paths)
(Optional later) CUDA C/C++ → Common → Use Fast Math: No (start accurate; turn Yes after validation)
Disable tricky link-time opts (if needed)
If you hit LNK2001/LTO issues: Linker → Optimization → Whole Program Optimization (/GL) → Disabled (Release only).
Sanity test (tiny ping)
In md_cuda.cu:
 #include <cuda_runtime.h>
__global__ void ping() {}
extern "C" bool md_cuda_ping() { ping<<<1,1>>>(); return cudaPeekAtLastError()==cudaSuccess; }
In md_cuda_iface.h:
 extern "C" bool md_cuda_ping();
Call md_cuda_ping() from main() (or your class). Build & run → should return true.
Nsight tools (optional now)
Install Nsight Systems/Compute via CUDA installer. In VS: Extensions → Nsight to profile later.
