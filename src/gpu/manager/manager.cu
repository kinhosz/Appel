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

    size_t size;
    
    size = maxTriangles * sizeof(float);
    host_res_dist = (float*)malloc(size);
    CUDA_STATUS(cudaMalloc((void**)&dvc_res_dist, size));

    CUDA_STATUS(cudaMalloc((void**)&blocks_N, sizeof(int)));
    CUDA_STATUS(cudaMalloc((void**)&triangles_N, sizeof(int)));
    CUDA_STATUS(cudaMalloc((void**)&rays_N, sizeof(int)));

    int blocks = maxTriangles / threadsperblock_y;

    size = blocks * sizeof(int);
    CUDA_STATUS(cudaMalloc((void**)&dvc_buffer_idx, size));
    host_buffer_idx = (int*)malloc(size);

    size = blocks * sizeof(float);
    CUDA_STATUS(cudaMalloc((void**)&dvc_buffer_dist, size));
    host_buffer_dist = (float*)malloc(size);

    /* allocate for cache and host_cache */
    aux = (GTriangleArray*)malloc(sizeof(GTriangleArray));
    host_cache = (GTriangleArray*)malloc(sizeof(GTriangleArray));
    CUDA_STATUS(cudaMalloc((void**)&cache, sizeof(GTriangleArray)));

    size = maxTriangles * sizeof(int);
    host_cache->host_id = (int*)malloc(size);
    CUDA_STATUS(cudaMalloc((void**)&aux->host_id, size));

    size = maxTriangles * sizeof(float);
    for(int i=0;i<3;i++) {
        host_cache->point[i].x = (float*)malloc(size);
        CUDA_STATUS(cudaMalloc((void**)&(aux->point[i].x), size));

        host_cache->point[i].y = (float*)malloc(size);
        CUDA_STATUS(cudaMalloc((void**)&(aux->point[i].y), size));

        host_cache->point[i].z = (float*)malloc(size);
        CUDA_STATUS(cudaMalloc((void**)&(aux->point[i].z), size));
    }

    CUDA_STATUS(cudaMemcpy(cache, aux, sizeof(GTriangleArray), cudaMemcpyHostToDevice));
    /* end of allocate */

    size = BATCHSIZE * sizeof(GRay);
    CUDA_STATUS(cudaMalloc((void**)&dvc_rays, size));
    host_rays = (GRay*)malloc(size);

    CUDA_STATUS(cudaDeviceSynchronize());

    CUDA_STATUS(cudaMemcpy(triangles_N, &maxTriangles, sizeof(int), 
        cudaMemcpyHostToDevice));

    CUDA_STATUS(cudaDeviceSynchronize());

    for(int i=0;i<maxTriangles;i++) {
        GTriangle gt;
        
        host_cache->host_id[i] = -1;
        for(int j=0;j<3;j++) {
            host_cache->point[j].x[i] = gt.point[j].x;
            host_cache->point[j].y[i] = gt.point[j].y;
            host_cache->point[j].z[i] = gt.point[j].z;
        }
    }

    free_pos = 0;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d: %s\n", device, deviceProp.name);
        printf("Maximum Streams: %d\n", deviceProp.asyncEngineCount);
    }

    minClock = __INT_MAX__;
    maxClock = -1;
    avgClock = 0;
    cntCalls = 0;
}

Manager::~Manager() {
    CUDA_STATUS(cudaFree(blocks_N));
    CUDA_STATUS(cudaFree(triangles_N));
    CUDA_STATUS(cudaFree(rays_N));
    CUDA_STATUS(cudaFree(dvc_rays));
    free(host_rays);
    free(host_res_dist);
    CUDA_STATUS(cudaFree(dvc_res_dist));
    CUDA_STATUS(cudaFree(dvc_buffer_idx));
    free(host_buffer_idx);
    CUDA_STATUS(cudaFree(dvc_buffer_dist));
    free(host_buffer_dist);

    CUDA_STATUS(cudaMemcpy(aux, cache, sizeof(GTriangleArray), cudaMemcpyDeviceToHost));
    for(int i=0;i<3;i++) {
        CUDA_STATUS(cudaFree(aux->point[i].x));
        CUDA_STATUS(cudaFree(aux->point[i].y));
        CUDA_STATUS(cudaFree(aux->point[i].z));

        free(host_cache->point[i].x);
        free(host_cache->point[i].y);
        free(host_cache->point[i].z);
    }

    CUDA_STATUS(cudaFree(aux->host_id));
    free(host_cache->host_id);

    CUDA_STATUS(cudaFree(cache));
    free(host_cache);
    free(aux);
}

#endif
