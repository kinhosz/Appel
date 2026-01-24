#ifndef APPEL_GPU_DISABLED

#include <Appel/gpu/manager.h>
#include <Appel/gpu/types/ray.h>
#include <Appel/gpu/kernel.h>
#include <Appel/gpu/pragma.h>
#include <Appel/gpu/reducer.h>

namespace Appel {
    int Manager::run(const Ray& ray) {
        GRay gr(ray);

        dvc_N[0] = free_pos;
        CUDA_STATUS(cudaMemcpy(cache, tmp, free_pos * sizeof(GTriangle), cudaMemcpyHostToDevice));

        CUDA_STATUS(cudaDeviceSynchronize());
        *dvc_ray = gr;

        free_pos = 0;

        castRay<<<blocksPerGrid, threadsPerBlock>>>(dvc_ray, buffer_dist, buffer_idx, cache, dvc_N);
        CUDA_STATUS(cudaDeviceSynchronize());

        getMin<<<1,1>>>(buffer_dist, buffer_idx, dvc_BLOCK, result);
        CUDA_STATUS(cudaDeviceSynchronize());

        int host_id = -2;
        host_id = *result;

        return host_id;
    }
}

#endif
