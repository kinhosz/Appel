#ifndef PRAGMA_GPU_H
#define PRAGMA_GPU_H

#include <iostream>

#define CUDA_STATUS(XXX) \
    do { \
        if (XXX != cudaSuccess) { \
            std::cerr << "cuda error: " << cudaGetErrorString(XXX); \
            std::cerr << ". File: " << __FILE__; \
            std::cerr << " at line: " << __LINE__ << "\n"; \
            assert(false); \
        } \
    } while (0)

#endif
