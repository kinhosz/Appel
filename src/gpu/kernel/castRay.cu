#include <gpu/kernel.h>
#include <gpu/helper.h>
#include <stdio.h>

__global__ void castRay(GRay *rays, int *rays_N, GTriangle *cache, int *triangles_N, 
    float *buffer_dist, int *buffer_idx, int *buffer_N) {
    __shared__ float s_dist[1024];
    __shared__ int s_index[1024];

    int N = buffer_N[0];

    s_index[threadIdx.y] = -1;

    int ray_id = blockIdx.x;
    int triangle_id = threadIdx.y + blockDim.y * blockIdx.y;

    if(threadIdx.y == 0 && ray_id < rays_N[0]) {        
        buffer_idx[ray_id * N + blockIdx.y] = -1;
    }

    const GTriangle triangle = cache[triangle_id];
    const GRay ray = rays[ray_id];

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        buffer_idx[ray_id * N + blockIdx.y] = -1;
    }

    __syncthreads();

    if(ray_id >= rays_N[0] || triangle_id >= triangles_N[0]) return;

    float d = triangleIntersect(ray, triangle);
    if(f_cmp(d, 0.00) == 1) {
        s_index[threadIdx.y] = triangle.host_id;
        s_dist[threadIdx.y] = d;
    }

    __syncthreads();

    int ts = 1024;

    int ty = threadIdx.y;

    while(ts > 1) {
        if(ty >= ts/2) return;

        if(s_index[ty] == -1 || (s_index[ty + ts/2] != -1 && f_cmp(s_dist[ty], s_dist[ty + ts/2]) == 1)) {
            s_index[ty] = s_index[ty+ts/2];
            s_dist[ty] = s_dist[ty+ts/2];
        }

        ts /= 2;

        __syncthreads();
    }

    buffer_idx[ray_id * N + blockIdx.y] = s_index[0];
    buffer_dist[ray_id * N + blockIdx.y] = s_dist[0];
}
