#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/types/ray.h>
#include <gpu/kernel.h>
#include <gpu/pragma.h>

int Manager::run(const Ray& ray) {
    GRay gr(ray);

    int threadsperblock = 1024;

    castRay<<<1,threadsperblock>>>(gr, buffer, cache, (int)maxTriangles);
    CUDA_STATUS(cudaDeviceSynchronize());

    int host_id = -2;
    CUDA_STATUS(cudaMemcpy(&host_id, buffer, sizeof(int),
        cudaMemcpyDeviceToHost));

    return host_id;
}

#endif
