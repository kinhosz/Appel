#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/types/ray.h>
#include <gpu/kernel.h>
#include <gpu/pragma.h>
#include <gpu/reducer.h>

std::vector<int> Manager::run(const std::vector<Ray> &rays) {
    int blocks = ((free_pos + threadsperblock_y - 1) / threadsperblock_y);
    free_pos = blocks * threadsperblock_y;

    for(int i=0;i<(int)rays.size();i++) {
        host_rays[i] = GRay(rays[i]);
    }

    size_t size;

    /* memcpy GTriangleArray */
    size = free_pos * sizeof(float);
    for(int j=0;j<3;j++) {
        CUDA_STATUS(cudaMemcpy(aux->point[j].x, host_cache->point[j].x, size, cudaMemcpyHostToDevice));
        CUDA_STATUS(cudaMemcpy(aux->point[j].y, host_cache->point[j].y, size, cudaMemcpyHostToDevice));
        CUDA_STATUS(cudaMemcpy(aux->point[j].z, host_cache->point[j].z, size, cudaMemcpyHostToDevice));
    }
    size = free_pos * sizeof(int);
    CUDA_STATUS(cudaMemcpy(aux->host_id, host_cache->host_id, size, cudaMemcpyHostToDevice));

    CUDA_STATUS(cudaMemcpy(cache, aux, sizeof(GTriangleArray), cudaMemcpyHostToDevice));

    /* end of memcpy */

    int h_rays_N = (int)rays.size();
    CUDA_STATUS(cudaMemcpy(rays_N, &h_rays_N, sizeof(int), cudaMemcpyHostToDevice));

    size = h_rays_N * sizeof(GRay);
    CUDA_STATUS(cudaMemcpy(dvc_rays, host_rays, size, cudaMemcpyHostToDevice));

    dim3 threadsperblock(threadsperblock_x, threadsperblock_y);

    int grid_x = h_rays_N;
    int grid_y = blocks;

    dim3 grids(grid_x, grid_y);

    CUDA_STATUS(cudaMemcpy(triangles_N, &free_pos, sizeof(int), cudaMemcpyHostToDevice));

    /*profiling*/
    int *h_table = (int*)malloc((blocks * 32) * sizeof(int));
    int *d_table;
    CUDA_STATUS(cudaMalloc((void**)&d_table, (blocks * 32) * sizeof(int)));

    CUDA_STATUS(cudaDeviceSynchronize());

    castRay<<<grids, threadsperblock>>>(dvc_rays, rays_N, cache, triangles_N, dvc_res_dist, d_table);
    CUDA_STATUS(cudaDeviceSynchronize());

    CUDA_STATUS(cudaMemcpy(h_table, d_table, (blocks * 32) * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i=0; i<(blocks*32); i++) {
        minClock = min(minClock, h_table[i]);
        maxClock = max(maxClock, h_table[i]);
        avgClock += h_table[i];
        cntCalls++;
    }

    //printf("min[%d]. max[%d]. avg[%f]\n", minClock, maxClock, avgClock/cntCalls);

    getMin<<<1, threadsperblock_y>>>(cache, dvc_res_dist, dvc_buffer_idx, dvc_buffer_dist, triangles_N);
    CUDA_STATUS(cudaDeviceSynchronize());

    size = (blocks * sizeof(int));

    CUDA_STATUS(cudaMemcpy(host_buffer_idx, dvc_buffer_idx, size, cudaMemcpyDeviceToHost));

    size = (blocks * sizeof(float));
    CUDA_STATUS(cudaMemcpy(host_buffer_dist, dvc_buffer_dist, size, cudaMemcpyDeviceToHost));
    
    CUDA_STATUS(cudaDeviceSynchronize());

    std::vector<int> res;
    for(int i=0;i<(int)rays.size();i++) {
        res.push_back(host_buffer_idx[i]);
    }

    free_pos = 0;

    return res;
}

#endif
