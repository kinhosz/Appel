#include <gpu/kernel.h>
#include <gpu/helper.h>

__global__ void castRay(GRay ray, int* buffer, GTriangle* cache, int N) {
    __shared__ float dist[1200];
    __shared__ int idx[1200];

    int tid = threadIdx.x;

    dist[tid] = -1.0;
    idx[tid] = -1;

    int curr_id = tid;
    while(curr_id < N) {
        float d = triangleIntersect(ray, cache[curr_id]);
        if(d > 0.0) {
            if(idx[tid] == -1 || dist[tid] > d) {
                idx[tid] = cache[curr_id].host_id;
                dist[tid] = d;
            }
        }

        curr_id += blockDim.x;
    }

    __syncthreads();

    if(tid != 0) return;

    int sz = min(N, blockDim.x);

    for(int i=1;i<sz;i++) {
        if(idx[i] == -1) continue;
        if(idx[0] == -1 || dist[0] > dist[i]) {
            idx[0] = idx[i];
            dist[0] = dist[i];
        }
    }

    *buffer = idx[0];
}
