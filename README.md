# Simple Molecular Dynamics (DSPC Project)

This repository contains a C++ Molecular Dynamics (MD) simulation with both CPU and GPU backends. The GPU implementation is built with CUDA 12.9, targeting NVIDIA RTX 30-series GPUs (specifically RTX 3050). The primary goal of the GPU backend is to build Verlet neighbor lists and compute Lennard-Jones (LJ) forces using double precision, with one thread per atom and no atomic operations.

---

## 📂 Repository Directory

```text
DSPC/
├── SimpleMD/
│   ├── SimpleMolecularDynamics/
│   │   ├── AndersonThermostat.cc / .h
│   │   ├── bench_md.cpp                # Headless benchmarking (main)
│   │   ├── example.cc
│   │   ├── example_2d.cc               # MD simulation display & execution (main)
│   │   ├── mat.cc / .h
│   │   ├── md_cuda.cu                  # GPU kernels + device code
│   │   ├── md_cuda_iface.cpp           # Thin host wrappers calling CUDA
│   │   ├── md_cuda_iface.h             # Host-visible API
│   │   ├── MDParticle.h
│   │   ├── MDParticleList.h
│   │   ├── MDParticleListEntry.h
│   │   ├── MDSim.cc / .h               # CPU simulation logic (Contains hotspots)
│   │   └── vec.cc / .h
│   └── x64/
│       ├── Debug/
│       ├── Release-CPU/
│       └── Release-GPU/
└── README.md
```

---

## 🛠️ Project Setup (Windows + VS2022)

To successfully build and run this project, follow these configuration steps in Microsoft Visual Studio 2022.

### 1. Initial Setup & Enabling CUDA
1. Double-click on the solution file: `DSPC/SimpleMD/SimpleMolecularDynamics.sln`.
2. Right-click your project in the Solution Explorer → **Build Customizations...** → Tick **CUDA 12.9** → **OK**.
3. Ensure the active Solution Platform is set to **x64** (not Win32).

### 2. Add CUDA Files
Ensure the following files are added to the project. Visual Studio will auto-detect `.cu` as CUDA C/C++ once the build customization is enabled:
* `md_cuda.cu` (Kernels and device code)
* `md_cuda_iface.h` (Host-visible API)
* `md_cuda_iface.cpp` (Thin wrappers calling CUDA)

### 3. Project Properties (Release & Debug)
Right-click your project → **Properties**. Apply the following settings:

* **C/C++ → Language:** C++ Language Standard: `ISO C++17`
* **C/C++ → Code Generation:** Runtime Library: `Multi-threaded DLL (/MD)`
* **C/C++ → Preprocessor:** Preprocessor Definitions: Add `USE_CUDA` (allows you to `#ifdef GPU` paths)
* **CUDA C/C++ → Device:** Code Generation: `compute_86,sm_86` (Targeted for RTX 3050)
* **CUDA C/C++ → Common:** Generate Relocatable Device Code: `Yes (-rdc=true)`
* **CUDA C/C++ → Common:** Use Fast Math: `No` *(Optional: Start accurate; switch to Yes after validation)*
* **CUDA Linker → General:** Perform Device Link: `Yes`

### 4. Create Mixed-Precision Configuration (Optional)
To test FP32 performance alongside double precision:
1. Right-click your project → **Configuration Manager...** → Active solution configuration → **<New...>**
2. **Name:** `Release-GPU-FP32`
3. **Copy settings from:** `Release-GPU`
4. With `Release-GPU-FP32` selected, go to Project Properties:
   * **C/C++ → Preprocessor:** Add `MD_MIXED_FP32`
   * **CUDA C/C++ → Preprocessor:** Add `MD_MIXED_FP32`

---

## 🧪 Sanity Test (Tiny Ping)

To verify your CUDA environment is configured correctly, implement a tiny ping test.

**In `md_cuda.cu`:**
```cpp
#include <cuda_runtime.h>

__global__ void ping() {}

extern "C" bool md_cuda_ping() {
    ping<<<1,1>>>();
    return cudaPeekAtLastError() == cudaSuccess;
}
```

**In `md_cuda_iface.h`:**
```cpp
extern "C" bool md_cuda_ping();
```
Call `md_cuda_ping()` from your `main()` or class. Build and run; it should return `true`.

---

## 🚀 Building and Running

### Important Build Note
> **⚠️ WARNING:** Only **one** `int main()` can be included in the build project at a time. Exclude the others by right-clicking the file in Solution Explorer and selecting **Exclude from Project** if you want to build successfully.

* **`bench_md.cpp`**: Use this for headless benchmarking. It will not display the MD simulation visually; it outputs time and potential energy for CPU/GPU comparisons.
* **`example_2d.cc`**: Use this for executing and displaying the live MD simulation visually.

### Headless Benchmarking Execution
To run the benchmarks, open the Visual Studio integrated terminal (`Ctrl` + `` ` ``) and execute the compiled binaries with the arguments `<Atoms> <Steps>`:

```powershell
# Run CPU Benchmark
.\x64\Release-CPU\SimpleMolecularDynamics.exe 1024 200

# Run GPU Benchmark
.\x64\Release-GPU\SimpleMolecularDynamics.exe 1024 200
```

---

## 🎯 Development Goals: CPU to GPU Offloading

Currently, `MDSim.cc` contains two computationally expensive hotspots caused by nested `for` loops. The main objective is to offload these to `md_cuda.cu`:

1. **Building Verlet Neighbor Lists**
2. **Computing Lennard-Jones (LJ) Forces** ```
