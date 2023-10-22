#include <gpu/kernel.h>
#include <gpu/helper.h>
#include <stdio.h>

__global__ void castRay(GRay *rays, int *rays_N, GTriangle *cache, int *triangles_N, 
    float *buffer_dist, int *buffer_idx, int *buffer_N) {
    __shared__ GTriangle s_cache[32];
    __shared__ GRay s_rays[32];
    __shared__ float s_dist[32][32];
    __shared__ int s_index[32][32];

    int N = buffer_N[0];

    s_index[threadIdx.x][threadIdx.y] = -1;

    int ray_id = threadIdx.x + blockDim.x * blockIdx.x;
    int triangle_id = threadIdx.y + blockDim.y * blockIdx.y;

    if(threadIdx.y == 0 && ray_id < rays_N[0]) {        
        buffer_idx[ray_id * N + blockIdx.y] = -1;
    }

    if(threadIdx.x == 0) {
        if(triangle_id < triangles_N[0]) {
            s_cache[threadIdx.y] = cache[triangle_id];
        }
    }

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        buffer_idx[ray_id * N + blockIdx.y] = -1;
    }

    if(threadIdx.y == 0) {
        if(ray_id < rays_N[0]) {
            s_rays[threadIdx.x] = rays[ray_id];
        }
    }

    __syncthreads();

    if(ray_id >= rays_N[0] || triangle_id >= triangles_N[0]) return;

    float d = triangleIntersect(s_rays[threadIdx.x], s_cache[threadIdx.y]);
    if(f_cmp(d, 0.00) == 1) {
        s_index[threadIdx.x][threadIdx.y] = s_cache[threadIdx.y].host_id;
        s_dist[threadIdx.x][threadIdx.y] = d;
    }

    __syncthreads();

    int ts = 32;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    while(ts > 1) {
        if(ty >= ts/2) return;

        if(s_index[tx][ty] == -1 || (s_index[tx][ty + ts/2] != -1 && f_cmp(s_dist[tx][ty], s_dist[tx][ty + ts/2]) == 1)) {
            s_index[tx][ty] = s_index[tx][ty+ts/2];
            s_dist[tx][ty] = s_dist[tx][ty+ts/2];
        }

        ts /= 2;

        __syncthreads();
    }

    buffer_idx[ray_id * N + blockIdx.y] = s_index[tx][0];
    buffer_dist[ray_id * N + blockIdx.y] = s_dist[tx][0];
}
