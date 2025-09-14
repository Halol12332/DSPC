#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include "md_cuda_iface.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(_e));                                \
        return false;                                                   \
    } } while(0)
#endif

// ---------- sanity ----------
__global__ void ping_kernel() {}
extern "C" bool md_cuda_ping() { ping_kernel << <1, 1 >> > (); return cudaDeviceSynchronize() == cudaSuccess; }

// ---------- helpers ----------
__device__ inline double pbc_min_image(double d, double L) {
    d -= nearbyint(d / L) * L;  // min image
    return d;
}
__device__ inline double lj_force_scalar(double r2, double eps, double sig) {
    // 24*eps*(2*(sig^12)/r^14 - (sig^6)/r^8) but we return factor to multiply by dx,dy,dz
    double inv2 = 1.0 / r2;
    double sr2 = (sig * sig) * inv2;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    return 24.0 * eps * (2.0 * sr12 - sr6) * inv2; // |F|/r * r̂ components multiply by dx,dy,dz
}
// float versions for mixed-precision path
__device__ inline float pbc_min_image_f(float d, float L) {
    d -= nearbyintf(d / L) * L;  // min image
    return d;
}
__device__ inline float lj_force_scalar_f(float r2, float eps, float sig) {
    // 24*eps*(2*(sig^12)/r^14 - (sig^6)/r^8), returned as factor to multiply by dx,dy,dz, in float
    float inv2 = 1.0f / r2;
    float sr2 = (sig * sig) * inv2;
    float sr6 = sr2 * sr2 * sr2;
    float sr12 = sr6 * sr6;
    return 24.0f * eps * (2.0f * sr12 - sr6) * inv2;
}


// ---------- kernels ----------
__global__ void kernel_cell_ids(const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    int N, double Lx, double Ly, double Lz,
    double cellL, int nx, int ny, int nz,
    int* __restrict__ cellId,
    int* __restrict__ atomId)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    auto wrap_index = [](int c, int n)->int { c %= n; return (c < 0) ? c + n : c; };
    int icx = wrap_index((int)floor((x[i] + 0.5 * Lx) / cellL), nx);
    int icy = wrap_index((int)floor((y[i] + 0.5 * Ly) / cellL), ny);
    int icz = wrap_index((int)floor((z[i] + 0.5 * Lz) / cellL), nz);
    int cid = (icz * ny + icy) * nx + icx;
    cellId[i] = cid;
    atomId[i] = i;
}

__global__ void kernel_cell_ranges(const int* __restrict__ cellId_sorted,
    int N, int numCells,
    int* __restrict__ cellStart,
    int* __restrict__ cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cid = cellId_sorted[i];

    if (i == 0 || cellId_sorted[i - 1] != cid) cellStart[cid] = i;
    if (i == N - 1 || cellId_sorted[i + 1] != cid) cellEnd[cid] = i + 1;
}

// ---------- safe wrap_index ----------
__device__ __forceinline__ int wrap_index(int c, int n) {
    c %= n;
    return (c < 0) ? c + n : c;
}

// ---------- safe kernel_build_neighbors ----------
__global__ void kernel_build_neighbors_safe(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    int N, double Lx, double Ly, double Lz, int dim,
    const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
    const int* __restrict__ atom_sorted, const int* __restrict__ cellId_sorted,
    int nx, int ny, int nz, double cellL, double rlist2, bool periodic,
    int maxNeigh, int* __restrict__ neighIdx, int* __restrict__ neighCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // recompute cell indices safely
    int icx = wrap_index((int)floor((x[i] + 0.5 * Lx) / cellL), nx);
    int icy = wrap_index((int)floor((y[i] + 0.5 * Ly) / cellL), ny);
    int icz = wrap_index((int)floor((z[i] + 0.5 * Lz) / cellL), nz);

    int base = i * maxNeigh;
    int count = 0;

    // iterate over 27 neighboring cells
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int cx = wrap_index(icx + dx, nx);
                int cy = wrap_index(icy + dy, ny);
                int cz = wrap_index(icz + dz, nz);
                int cid = (cz * ny + cy) * nx + cx;

                int start = cellStart[cid];
                int end = cellEnd[cid];

                if (start < 0 || end <= 0) continue;

                for (int p = start; p < end; ++p) {
                    int j = atom_sorted[p];
                    if (j == i) continue;

                    double dx = x[j] - x[i];
                    double dy = y[j] - y[i];
                    double dz = z[j] - z[i];

                    if (periodic) {
                        dx = dx - nearbyint(dx / Lx) * Lx;
                        if (dim >= 2) dy = dy - nearbyint(dy / Ly) * Ly; else dy = 0.0;
                        if (dim >= 3) dz = dz - nearbyint(dz / Lz) * Lz; else dz = 0.0;
                    }
                    else {
                        if (dim < 2) dy = 0.0;
                        if (dim < 3) dz = 0.0;
                    }

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 <= rlist2) {
                        if (count < maxNeigh) {
                            neighIdx[base + count] = j;
                            ++count;
                        }
                        else {
                            // safely skip excess neighbors
                            // optionally: atomicAdd a counter to detect overflows
                        }
                    }
                }
            }
        }
    }

    neighCount[i] = count;
}


__global__ void kernel_lj_forces(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    double* __restrict__ fx, double* __restrict__ fy, double* __restrict__ fz,
    const int* __restrict__ neighIdx, const int* __restrict__ neighCount,
    int N, double Lx, double Ly, double Lz, int dim, double rc2, bool periodic,
    double eps, double sigma, double* __restrict__ potOut, int maxNeigh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

#ifdef MD_MIXED_FP32
    // --------- Mixed-precision path: float math, double accumulation ---------
    float xi = (float)x[i], yi = (float)y[i], zi = (float)z[i];
    double fxi = 0.0, fyi = 0.0, fzi = 0.0, poti = 0.0;

    int base = i * maxNeigh;
    int nbh = neighCount[i];

    float Lx_f = (float)Lx, Ly_f = (float)Ly, Lz_f = (float)Lz;
    float eps_f = (float)eps, sigma_f = (float)sigma, rc2_f = (float)rc2;

    for (int k = 0; k < nbh; ++k) {
        int j = neighIdx[base + k];
        float dx = (float)x[j] - xi;
        float dy = (float)y[j] - yi;
        float dz = (float)z[j] - zi;

        if (periodic) {
            dx = pbc_min_image_f(dx, Lx_f);
            if (dim >= 2) dy = pbc_min_image_f(dy, Ly_f); else dy = 0.0f;
            if (dim >= 3) dz = pbc_min_image_f(dz, Lz_f); else dz = 0.0f;
        }
        else {
            if (dim < 3) dz = 0.0f;
            if (dim < 2) dy = 0.0f;
        }

        float r2f = dx * dx + dy * dy + dz * dz;
        if (r2f == 0.0f || (rc2_f > 0.0f && r2f > rc2_f)) continue;

        // force (float), accumulate (double)
        float fsc = lj_force_scalar_f(r2f, eps_f, sigma_f);
        fxi += (double)(fsc * dx);
        fyi += (double)(fsc * dy);
        fzi += (double)(fsc * dz);

        // potential (float), accumulate (double, with 0.5 to avoid double counting)
        float inv2 = 1.0f / r2f;
        float sr2 = (sigma_f * sigma_f) * inv2;
        float sr6 = sr2 * sr2 * sr2;
        float sr12 = sr6 * sr6;
        float vij = 4.0f * eps_f * (sr12 - sr6);
        poti += 0.5 * (double)vij;
    }

    fx[i] = fxi; fy[i] = fyi; fz[i] = fzi;
    potOut[i] = poti;

#else
    // --------- Original double-precision path (unchanged) ---------
    double xi = x[i], yi = y[i], zi = z[i];
    double fxi = 0.0, fyi = 0.0, fzi = 0.0;
    double poti = 0.0;

    int base = i * maxNeigh;
    int nbh = neighCount[i];

    for (int k = 0; k < nbh; ++k) {
        int j = neighIdx[base + k];
        double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
        if (periodic) {
            dx = pbc_min_image(dx, Lx);
            if (dim >= 2) dy = pbc_min_image(dy, Ly); else dy = 0.0;
            if (dim >= 3) dz = pbc_min_image(dz, Lz); else dz = 0.0;
        }
        else {
            if (dim < 3) dz = 0.0;
            if (dim < 2) dy = 0.0;
        }
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 == 0.0 || (rc2 > 0.0 && r2 > rc2)) continue;

        double fsc = lj_force_scalar(r2, eps, sigma);
        fxi += fsc * dx; fyi += fsc * dy; fzi += fsc * dz;

        double inv2 = 1.0 / r2;
        double sr2 = (sigma * sigma) * inv2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double vij = 4.0 * eps * (sr12 - sr6);
        poti += 0.5 * vij;
    }

    fx[i] = fxi; fy[i] = fyi; fz[i] = fzi;
    potOut[i] = poti;
#endif
}


extern "C" bool md_cuda_init(DevMD* d, int N, int dim, const double boxL[3],
    int nx, int ny, int nz, double cellL, int maxNeigh)
{
    if (!d) return false;
    d->N = N; d->dim = dim; d->boxL[0] = boxL[0]; d->boxL[1] = boxL[1]; d->boxL[2] = boxL[2];
    d->nx = nx; d->ny = ny; d->nz = nz; d->numCells = nx * ny * nz; d->cellL = cellL; d->maxNeigh = maxNeigh;

    size_t Nd = (size_t)N, Nc = (size_t)d->numCells;
    CUDA_CHECK(cudaMalloc(&d->x, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d->y, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d->z, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d->fx, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d->fy, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d->fz, Nd * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d->cellId_unsorted, Nd * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->atom_unsorted, Nd * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->cellId_sorted, Nd * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->atom_sorted, Nd * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->cellStart, Nc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->cellEnd, Nc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->neighIdx, Nd * maxNeigh * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->neighCount, Nd * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d->potScratch, Nd * sizeof(double)));
    CUDA_CHECK(cudaMemset(d->fx, 0, Nd * sizeof(double)));
    CUDA_CHECK(cudaMemset(d->fy, 0, Nd * sizeof(double)));
    CUDA_CHECK(cudaMemset(d->fz, 0, Nd * sizeof(double)));
    return true;
}

extern "C" void md_cuda_free(DevMD* d) {
    if (!d) return;
    auto F = [&](void* p) { if (p) cudaFree(p); };
    F(d->x); F(d->y); F(d->z); F(d->fx); F(d->fy); F(d->fz);
    F(d->cellId_unsorted); F(d->atom_unsorted);
    F(d->cellId_sorted); F(d->atom_sorted);
    F(d->cellStart); F(d->cellEnd);
    F(d->neighIdx); F(d->neighCount);
    F(d->potScratch);
    *d = DevMD{};
}

extern "C" bool md_cuda_upload_pos(DevMD* d, const double* xh, const double* yh, const double* zh) {
    if (!d) return false;
    size_t b = (size_t)d->N * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d->x, xh, b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->y, yh, b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->z, zh, b, cudaMemcpyHostToDevice));
    return true;
}
extern "C" bool md_cuda_download_force(DevMD* d, double* fxh, double* fyh, double* fzh) {
    if (!d) return false;
    size_t b = (size_t)d->N * sizeof(double);
    CUDA_CHECK(cudaMemcpy(fxh, d->fx, b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fyh, d->fy, b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fzh, d->fz, b, cudaMemcpyDeviceToHost));
    return true;
}

extern "C" bool md_cuda_build_cell_ids(DevMD* d) {
    if (!d) return false;
    int B = 256, G = (d->N + B - 1) / B;
    kernel_cell_ids << <G, B >> > (d->x, d->y, d->z, d->N, d->boxL[0], d->boxL[1], d->boxL[2],
        d->cellL, d->nx, d->ny, d->nz,
        d->cellId_unsorted, d->atom_unsorted);
    return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" bool md_cuda_sort_by_cell(DevMD* d) {
    if (!d) return false;
    // copy unsorted -> sorted, then sort-by-key in-place
    size_t b = (size_t)d->N * sizeof(int);
    CUDA_CHECK(cudaMemcpy(d->cellId_sorted, d->cellId_unsorted, b, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d->atom_sorted, d->atom_unsorted, b, cudaMemcpyDeviceToDevice));
    thrust::device_ptr<int> k(d->cellId_sorted);
    thrust::device_ptr<int> v(d->atom_sorted);
    thrust::sort_by_key(k, k + d->N, v);
    return true;
}

extern "C" bool md_cuda_build_cell_ranges(DevMD* d) {
    if (!d) return false;
    // init to -1 (0xFF)
    CUDA_CHECK(cudaMemset(d->cellStart, 0xFF, (size_t)d->numCells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d->cellEnd, 0xFF, (size_t)d->numCells * sizeof(int)));
    int B = 256, G = (d->N + B - 1) / B;
    kernel_cell_ranges << <G, B >> > (d->cellId_sorted, d->N, d->numCells, d->cellStart, d->cellEnd);
    return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" bool md_cuda_build_neighbors(DevMD* d, double rlist, bool periodic) {
    if (!d) return false;
    int B = 256, G = (d->N + B - 1) / B;
    double rlist2 = (rlist > 0.0) ? rlist * rlist : 1e300;
    kernel_build_neighbors_safe << <G, B >> > (
        d->x, d->y, d->z, d->N, d->boxL[0], d->boxL[1], d->boxL[2], d->dim,
        d->cellStart, d->cellEnd, d->atom_sorted, d->cellId_sorted,
        d->nx, d->ny, d->nz, d->cellL, rlist2, periodic,
        d->maxNeigh, d->neighIdx, d->neighCount
        );
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" double md_cuda_forces(DevMD* d, double rc, double eps, double sigma, bool periodic) {
    if (!d) return 0.0;
    int B = 256, G = (d->N + B - 1) / B;
    double rc2 = (rc > 0.0) ? rc * rc : 1e300;
    kernel_lj_forces << <G, B >> > (
        d->x, d->y, d->z, d->fx, d->fy, d->fz,
        d->neighIdx, d->neighCount,
        d->N, d->boxL[0], d->boxL[1], d->boxL[2], d->dim, rc2, periodic,
        eps, sigma, d->potScratch, d->maxNeigh);
    cudaDeviceSynchronize();

    thrust::device_ptr<double> p(d->potScratch);
    return thrust::reduce(p, p + d->N, 0.0, thrust::plus<double>());
}
