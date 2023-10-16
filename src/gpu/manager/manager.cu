#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/pragma.h>
#include <gpu/types/triangle.h>
#include <gpu/kernel.h>

Manager::Manager(unsigned int maxTriangles, int height, int width,
    int depth, int maxLights) {

    this->height = height;
    this->width = width;
    this->depth = depth;
    this->maxLights = maxLights;

    this->free_light_pos = 0;

    this->triangles = maxTriangles;
    this->buffer_size = (1<<depth) * (maxLights + 1) * (height * width);

    size_t size = triangles * sizeof(GTriangle);
    CUDA_STATUS(cudaMalloc((void**)&cache_triangle, size));

    size = buffer_size * sizeof(int);
    CUDA_STATUS(cudaMalloc((void**)&device_buffer, size));
    host_buffer = (int *)malloc(size);

    size = (height * width) * sizeof(GRay);
    CUDA_STATUS(cudaMalloc((void**)&tmp_ray, size));

    size = maxLights * sizeof(GPoint);
    CUDA_STATUS(cudaMalloc((void**)&cache_light, size));

    for(int i=0;i<triangles;i++) {
        GTriangle t;
        t.host_id = -1;
        updateCacheTriangle<<<1,1>>>(i, t, cache_triangle);
    }
    for(int i=0;i<triangles;i++) free_pos.push(i);

    CUDA_STATUS(cudaDeviceSynchronize());
}

Manager::~Manager() {
    CUDA_STATUS(cudaFree(cache_triangle));
    CUDA_STATUS(cudaFree(device_buffer));
    free(host_buffer);

    CUDA_STATUS(cudaFree(tmp_ray));
    CUDA_STATUS(cudaFree(cache_light));
}

#endif
