#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/pragma.h>
#include <gpu/kernel.h>

void Manager::run(const Vetor &up, const Vetor &right, 
    const Vetor &front, const Vetor &location, double dist) {

    int pixels = height * width;
    int threadsperblocks = 1024;
    int blockspergrid = (pixels + threadsperblocks - 1)/threadsperblocks;

    CUDA_STATUS(cudaDeviceSynchronize());
    
    traceRayPreProcess<<<blockspergrid, threadsperblocks>>>(
        GPoint(up), GPoint(right), GPoint(front), GPoint(location),
        dist, height, width, cache_triangle, cache_light, depth, 
        this->free_light_pos, device_buffer);
    
    CUDA_STATUS(cudaDeviceSynchronize());

    size_t size = this->buffer_size * sizeof(int);
    CUDA_STATUS(cudaMemcpy(host_buffer, device_buffer, size, cudaMemcpyDeviceToHost));
}

#endif
