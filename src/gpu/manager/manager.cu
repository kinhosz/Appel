#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/pragma.h>
#include <gpu/types/triangle.h>
#include <gpu/kernel.h>
#include <gpu/types/ray.h>

Manager::Manager(unsigned int maxTriangles) {
    this->maxTriangles = maxTriangles;
    this->threadsPerBlock = 1024;
    this->blocksPerGrid = (maxTriangles + threadsPerBlock - 1) / threadsPerBlock;
    this->free_pos = 0;

    size_t size = maxTriangles * sizeof(GTriangle);
    CUDA_STATUS(cudaMalloc((void**)&cache, size));
    CUDA_STATUS(cudaMallocManaged((void**)&result, sizeof(int)));
    CUDA_STATUS(cudaMallocManaged((void**)&dvc_ray, sizeof(GRay)));
    CUDA_STATUS(cudaMallocManaged((void**)&dvc_N, sizeof(int)));
    CUDA_STATUS(cudaMallocManaged((void**)&dvc_BLOCK, sizeof(int)));
    
    size = blocksPerGrid * sizeof(float);
    CUDA_STATUS(cudaMalloc((void**)&buffer_dist, size));
    size = blocksPerGrid * sizeof(int);
    CUDA_STATUS(cudaMalloc((void**)&buffer_idx, size));

    size = maxTriangles * sizeof(GTriangle);
    tmp = (GTriangle* ) malloc(size);

    CUDA_STATUS(cudaDeviceSynchronize());

    dvc_N[0] = (int)maxTriangles;
    dvc_BLOCK[0] = blocksPerGrid;

    for(int i=0;i<(int)maxTriangles;i++) {
        GTriangle t;
        t.host_id = -1;
        updateCache<<<1,1>>>(i, t, cache);
    }

    CUDA_STATUS(cudaDeviceSynchronize());
}

Manager::~Manager() {
    CUDA_STATUS(cudaFree(result));
    CUDA_STATUS(cudaFree(cache));
    CUDA_STATUS(cudaFree(dvc_N));
    CUDA_STATUS(cudaFree(dvc_BLOCK));
    CUDA_STATUS(cudaFree(dvc_ray));
    CUDA_STATUS(cudaFree(buffer_dist));
    CUDA_STATUS(cudaFree(buffer_idx));
    free(tmp);
}

#endif
