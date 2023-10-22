#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/types/ray.h>
#include <gpu/kernel.h>
#include <gpu/pragma.h>
#include <gpu/reducer.h>

std::vector<int> Manager::run(const std::vector<Ray> &rays) {
    for(int i=0;i<(int)rays.size();i++) {
        host_rays[i] = GRay(rays[i]);
    }

    CUDA_STATUS(cudaDeviceSynchronize());

    int h_rays_N = (int)rays.size();
    CUDA_STATUS(cudaMemcpy(rays_N, &h_rays_N, sizeof(int), cudaMemcpyHostToDevice));

    size_t size = h_rays_N * sizeof(GRay);
    CUDA_STATUS(cudaMemcpy(dvc_rays, host_rays, size, cudaMemcpyHostToDevice));

    dim3 threadsperblock(threadsperblock_x, threadsperblock_y);
    
    int grid_x = (h_rays_N + threadsperblock_x - 1)/threadsperblock_x;
    int grid_y = (maxTriangles + threadsperblock_y - 1)/threadsperblock_y;

    dim3 grids(grid_x, grid_y);

    CUDA_STATUS(cudaDeviceSynchronize());

    castRay<<<grids, threadsperblock>>>(dvc_rays, rays_N, cache, triangles_N,
        buffer_dist, buffer_idx, dvc_bufferN);

    CUDA_STATUS(cudaDeviceSynchronize());

    int ts = 1024;
    int blocks = h_rays_N;

    getMin<<<blocks, ts>>>(buffer_dist, buffer_idx, rays_N, dvc_bufferN, dvc_res_idx);

    CUDA_STATUS(cudaDeviceSynchronize());

    size = ((int)rays.size()) * sizeof(int);

    CUDA_STATUS(cudaMemcpy(host_res_idx, dvc_res_idx, size, 
        cudaMemcpyDeviceToHost));
    
    std::vector<int> res;
    for(int i=0;i<(int)rays.size();i++) {
        res.push_back(host_res_idx[i]);
    }

    return res;
}

#endif
