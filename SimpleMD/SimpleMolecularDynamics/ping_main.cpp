#include <iostream>
#include "md_cuda_iface.h"
int main() {
#ifdef USE_CUDA
	bool ok = md_cuda_ping();
	std::cout << "[CUDA ping] " << (ok ? "OK" : "FAIL") << "\n";
#else
	std::cout << "[CUDA disabled] Build with USE_CUDA\n";
#endif
	return 0;
}
