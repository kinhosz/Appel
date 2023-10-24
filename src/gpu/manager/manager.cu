#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/pragma.h>
#include <gpu/types/triangle.h>
#include <gpu/kernel.h>
#include <gpu/types/ray.h>

Manager::Manager(int maxTriangles, int batchsize) {
    this->maxTriangles = maxTriangles;
    this->BATCHSIZE = batchsize;

    this->threadsperblock_x = 1;
    this->threadsperblock_y = 1024;

    this->bufferN = (maxTriangles + threadsperblock_y - 1) / threadsperblock_y;

    size_t size;

    CUDA_STATUS(cudaMalloc((void**)&dvc_bufferN, sizeof(int)));

    CUDA_STATUS(cudaMalloc((void**)&blocks_N, sizeof(int)));
    CUDA_STATUS(cudaMalloc((void**)&triangles_N, sizeof(int)));

    CUDA_STATUS(cudaMalloc((void**)&rays_N, sizeof(int)));

    size = maxTriangles * sizeof(GTriangle);
    CUDA_STATUS(cudaMalloc((void**)&cache, size));

    size = BATCHSIZE * sizeof(GRay);
    CUDA_STATUS(cudaMalloc((void**)&dvc_rays, size));
    host_rays = (GRay*)malloc(size);

    size = (BATCHSIZE * bufferN) * sizeof(int);
    CUDA_STATUS(cudaMalloc((void**)&buffer_idx, size));

    size = (BATCHSIZE * bufferN) * sizeof(float);
    CUDA_STATUS(cudaMalloc((void**)&buffer_dist, size));

    size = BATCHSIZE * sizeof(int);
    CUDA_STATUS(cudaMalloc((void**)&dvc_res_idx, size));
    host_res_idx = (int*)malloc(size);

    CUDA_STATUS(cudaDeviceSynchronize());

    CUDA_STATUS(cudaMemcpy(dvc_bufferN, &bufferN, sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_STATUS(cudaMemcpy(triangles_N, &maxTriangles, sizeof(int), 
        cudaMemcpyHostToDevice));

    CUDA_STATUS(cudaDeviceSynchronize());

    for(int i=0;i<maxTriangles;i++) {
        GTriangle gt;
        gt.host_id = -1;
        updateCache<<<1,1>>>(i, gt, cache);
        free_pos.push(i);
    }
}

Manager::~Manager() {
    CUDA_STATUS(cudaFree(blocks_N));
    CUDA_STATUS(cudaFree(triangles_N));
    CUDA_STATUS(cudaFree(rays_N));
    CUDA_STATUS(cudaFree(cache));
    CUDA_STATUS(cudaFree(dvc_rays));
    free(host_rays);
    CUDA_STATUS(cudaFree(dvc_res_idx));
    free(host_res_idx);

    CUDA_STATUS(cudaFree(buffer_idx));
    CUDA_STATUS(cudaFree(buffer_dist));
    CUDA_STATUS(cudaFree(dvc_bufferN));
}

#endif
