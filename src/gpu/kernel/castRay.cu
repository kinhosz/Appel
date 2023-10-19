#include <gpu/kernel.h>
#include <gpu/helper.h>

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

    if(tid != 0) return;

    for(int i=1;i<blockDim.x;i++) {
        if(idx[i] == -1) continue;
        if(idx[0] == -1 || dist[0] > dist[i]) {
            idx[0] = idx[i];
            dist[0] = dist[i];
        }
    }

    buffer_idx[blockIdx.x] = idx[0];
    buffer_dist[blockIdx.x] = dist[0];
}
