#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/pragma.h>
#include <gpu/types/triangle.h>
#include <gpu/kernel.h>

Manager::Manager(unsigned int maxTriangles) {
    this->maxTriangles = maxTriangles;

    size_t size = maxTriangles * sizeof(GTriangle);
    CUDA_STATUS(cudaMalloc((void**)&cache, size));
    CUDA_STATUS(cudaMalloc((void**)&buffer, sizeof(int)));
    CUDA_STATUS(cudaDeviceSynchronize());

    for(int i=0;i<(int)maxTriangles;i++) {
        GTriangle t;
        t.host_id = -1;
        updateCache<<<1,1>>>(i, t, cache);
        free_pos.push(i);
    }

    CUDA_STATUS(cudaDeviceSynchronize());
}

Manager::~Manager() {
    CUDA_STATUS(cudaFree(buffer));
    CUDA_STATUS(cudaFree(cache));
}

#endif
