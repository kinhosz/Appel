#include <Appel/gpu/kernel.h>
#include <Appel/gpu/helper.h>

__global__ void castRay(GRay* ray, float* buffer_dist, int* buffer_idx, GTriangle* cache, int* N) {
    __shared__ float dist[1200];
    __shared__ int idx[1200];

    int tid = threadIdx.x;
    int pointer = threadIdx.x + blockDim.x * blockIdx.x;

    dist[tid] = -1.0;
    idx[tid] = -1;

    if(tid == 0) {
        buffer_dist[blockIdx.x] = -1.0;
        buffer_idx[blockIdx.x] = -1;
    }

    if(pointer >= (*N)) return;

    float d = triangleIntersect(*ray, cache[pointer]);
    if(d > 0.0) {
        idx[tid] = cache[pointer].host_id;
        dist[tid] = d;
    }

    __syncthreads();
    
    int ts = blockDim.x;

    while(ts > 1) {
        if(tid >= ts/2) return;

        if(idx[tid] == -1 || (dist[tid] > dist[tid+ts/2] && idx[tid+ts/2] != -1)) {
            idx[tid] = idx[tid+ts/2];
            dist[tid] = dist[tid+ts/2];
        }

        ts /= 2;

        __syncthreads();
    }

    buffer_idx[blockIdx.x] = idx[0];
    buffer_dist[blockIdx.x] = dist[0];
}
