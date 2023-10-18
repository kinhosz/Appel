#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/types/ray.h>
#include <gpu/kernel.h>
#include <gpu/pragma.h>

int Manager::run(const Ray& ray) {
    GRay gr(ray);

    int threadsperblock = 1024;

    CUDA_STATUS(cudaDeviceSynchronize());

    *dvc_ray = gr;

    CUDA_STATUS(cudaDeviceSynchronize());
    castRay<<<1,threadsperblock>>>(dvc_ray, buffer, cache, dvc_N);
    CUDA_STATUS(cudaDeviceSynchronize());

    int host_id = -2;
    host_id = *buffer;

    return host_id;
}

#endif
