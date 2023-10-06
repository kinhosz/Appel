#include <gpu/types/triangle.h>
#include <gpu/types/ray.h>
#include <gpu/service.h>

__global__ void castRay(GTriangle *cache, GRay *ray, float *block_dist, int *block_idx, int N) {
    __shared__ int r_idx[1024];
    __shared__ float r_dist[1024];

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    r_dist[threadIdx.x] = -1.0;
    r_idx[threadIdx.x] = cache[thread_id].host_id;

    if(thread_id < N){
        if(cache[thread_id].host_id != -1) {

            float t = planeIntersect((*ray), cache[thread_id]);
            if(t > 0.0) {
                if(onTriangle(cache[thread_id], (*ray), t)) {
                    r_dist[threadIdx.x] = t;
                }
            }
        }
    }

    __syncthreads();

    if(threadIdx.x == 0) {
        float minT = MAXFLOAT;
        int idx = -1;

        for(int i=0;i<blockDim.x;i++) {
            if(r_dist[i] < 0.0) continue;
            if(r_dist[i] > minT) continue;

            minT = r_dist[i];
            idx = r_idx[i];
        }

        block_dist[blockIdx.x] = minT;
        block_idx[blockIdx.x] = idx;
    }
}
