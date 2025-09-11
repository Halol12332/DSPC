#pragma once
struct DevMD {
    int N{ 0 }, dim{ 3 };
    double boxL[3]{ 1.0,1.0,1.0 };
    double cellL{ 1.0 };
    int nx{ 1 }, ny{ 1 }, nz{ 1 }, numCells{ 1 };
    int maxNeigh{ 128 };
    // SoA
    double* x{ nullptr }, * y{ nullptr }, * z{ nullptr };
    double* fx{ nullptr }, * fy{ nullptr }, * fz{ nullptr };
    // cell/bin
    int* cellId_unsorted{ nullptr }, * atom_unsorted{ nullptr };
    int* cellId_sorted{ nullptr }, * atom_sorted{ nullptr };
    int* cellStart{ nullptr }, * cellEnd{ nullptr };  // numCells
    // neighbors
    int* neighIdx{ nullptr };    // N*maxNeigh
    int* neighCount{ nullptr };  // N
    // temp for potential reduction
    double* potScratch{ nullptr }; // N
};

#ifdef __cplusplus
extern "C" {
#endif
    bool md_cuda_init(DevMD* d, int N, int dim, const double boxL[3],
        int nx, int ny, int nz, double cellL, int maxNeigh);
    void md_cuda_free(DevMD* d);

    bool md_cuda_upload_pos(DevMD* d, const double* xh, const double* yh, const double* zh);
    bool md_cuda_download_force(DevMD* d, double* fxh, double* fyh, double* fzh);

    // pipeline steps
    bool md_cuda_build_cell_ids(DevMD* d);
    bool md_cuda_sort_by_cell(DevMD* d);
    bool md_cuda_build_cell_ranges(DevMD* d);
    bool md_cuda_build_neighbors(DevMD* d, double rlist, bool periodic);

    // forces (returns total potential)
    double md_cuda_forces(DevMD* d, double rc, double eps, double sigma, bool periodic);

    // sanity
    bool md_cuda_ping();
#ifdef __cplusplus
}
#endif
