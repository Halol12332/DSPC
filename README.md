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


SimpleMolecularDynamics — CPU vs CUDA Bench
0) Requirements

Windows, VS2022 (x64)

CUDA Toolkit 12.9

NVIDIA GPU (RTX 3050, SM 8.6)

1) Files added/edited

CUDA

md_cuda.cu – kernels (cell IDs, sort, cell ranges, neighbors, LJ forces)

md_cuda_iface.h/.cpp – host API (md_cuda_init/free, upload/download, pipeline)

Bench

bench_md.cpp – headless timing driver (no GLUT)

MDSim patches

Public wrappers in MDSim.h:

// Benchmark wrappers
double benchRefresh(bool c, bool r) { return refreshVerletLists(c, r); }
double benchForce() { return velocityVerletForce(); }


MDSim::refreshVerletLists (GPU path):

Upload positions only.

Do not build GPU neighbor list here (early-return when !calc && !countRadial).

MDSim::velocityVerletForce (GPU path):

Build neighbors once (IDs → sort → ranges → neighbors).

Launch LJ force kernel (1 thread/atom, symmetric list), copy forces back.

LJ params match CPU code: ε = 1.0, σ = 0.025.
Set maxNeigh = 256–512 in md_cuda_init(...) to avoid truncation.

2) Create two build configs

Build → Configuration Manager…

Add Release-CPU (copy from Release)

Add Release-GPU (copy from Release)

Platform: x64

Release-CPU (serial)

Project → Properties (Release-CPU | x64)

C/C++ → Preprocessor: remove USE_CUDA

Solution Explorer: md_cuda.cu → Exclude From Build

Keep only bench_md.cpp included (exclude other files with main())

Release-GPU (CUDA)

Project → Build Customizations… → tick CUDA 12.9

Project → Properties (Release-GPU | x64)

C/C++ → Preprocessor: add USE_CUDA

C/C++ → Language: C++17

CUDA C/C++ → Device → Code Generation: compute_86,sm_86

CUDA C/C++ → Common → Relocatable Device Code: Yes (-rdc=true)

CUDA Linker → General → Perform Device Link: Yes

Solution Explorer: md_cuda.cu → Include In Build

Ensure only bench_md.cpp provides main() (exclude others)

3) Build

CPU: Build → Rebuild Solution (Release-CPU | x64)

GPU: Build → Rebuild Solution (Release-GPU | x64)

(CLI alt)

msbuild SimpleMolecularDynamics.sln /t:Rebuild /p:Configuration="Release-CPU";Platform="x64"
msbuild SimpleMolecularDynamics.sln /t:Rebuild /p:Configuration="Release-GPU";Platform="x64"

4) Run the benchmark

From repo root (SimpleMD):

CPU

.\x64\Release-CPU\SimpleMolecularDynamics.exe 1024 200


GPU

.\x64\Release-GPU\SimpleMolecularDynamics.exe 1024 200


Output looks like:

[MODE] CPU N=1024 steps=200 time_ms=23030.5 pot=...
[MODE] GPU N=1024 steps=200 time_ms=  676.9 pot=...


Speedup = CPU_time_ms / GPU_time_ms (e.g., ≈ 34×).

5) Quick correctness check

Compare potentials on a tiny run:

.\x64\Release-CPU\SimpleMolecularDynamics.exe 256 1
.\x64\Release-GPU\SimpleMolecularDynamics.exe 256 1


If the difference > ~1–2%:

Increase maxNeigh to 512 in md_cuda_init(...).

Ensure r_verlet ≥ r_inter (e.g., 0.22 ≥ 0.15).

Confirm periodic minimum image is applied in both paths.

6) Optional: profile with Nsight Systems
nsys profile --stats=true --force-overwrite=true -o md_gpu ^
  .\x64\Release-GPU\SimpleMolecularDynamics.exe 1024 200


Open md_gpu.qdrep in Nsight Systems to inspect kernel/memcpy timing.

7) Troubleshooting (common)

GPU slower than CPU: ensure CPU list isn’t built on GPU path (refreshVerletLists early-return), build neighbors only in velocityVerletForce.

Multiple mains: exclude example_2d.cc, ping_main.cpp for bench builds.

C++17 warnings: set C++ to ISO C++17 for both host and CUDA.

“vector component index higher than vector dimension”: when copying forces back, write directly to p->a[1..dim] (avoid default-constructed vec).