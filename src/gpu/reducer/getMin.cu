#include <gpu/reducer.h>
#include <gpu/helper.h>
#include <stdio.h>

__device__ void warpReduce(volatile float *s_dist, volatile int *s_idx, int tx) {
    if(s_dist[tx] > s_dist[tx + 32]) {
        s_dist[tx] = s_dist[tx + 32];
        s_idx[tx] = s_idx[tx + 32];
    }
    if(s_dist[tx] > s_dist[tx + 16]) {
        s_dist[tx] = s_dist[tx + 16];
        s_idx[tx] = s_idx[tx + 16];
    }
    if(s_dist[tx] > s_dist[tx + 8]) {
        s_dist[tx] = s_dist[tx + 8];
        s_idx[tx] = s_idx[tx + 8];
    }
    if(s_dist[tx] > s_dist[tx + 4]) {
        s_dist[tx] = s_dist[tx + 4];
        s_idx[tx] = s_idx[tx + 4];
    }
    if(s_dist[tx] > s_dist[tx + 2]) {
        s_dist[tx] = s_dist[tx + 2];
        s_idx[tx] = s_idx[tx + 2];
    }
    if(s_dist[tx] > s_dist[tx + 1]) {
        s_dist[tx] = s_dist[tx + 1];
        s_idx[tx] = s_idx[tx + 1];
    }
}

__global__ void getMin(GTriangleArray *cache, float *dvc_res_dist, int *dvc_buffer_idx, float *dvc_buffer_dist, int *N) {
    __shared__ float s_dist[1024];
    __shared__ int s_idx[1024];

    int tx = threadIdx.x;
    int i = threadIdx.x + (blockDim.x * blockIdx.x);

    s_dist[tx] = dvc_res_dist[i];
    s_idx[tx] = cache->host_id[i];

    while(i + blockDim.x < N[0]) {
        if(s_dist[tx] > dvc_res_dist[i + blockDim.x]) {
            s_dist[tx] = dvc_res_dist[i + blockDim.x];
            s_idx[tx] = cache->host_id[i + blockDim.x];
        }

        i += blockDim.x;
    }
    __syncthreads();

    if(tx < 512) {
        if(s_dist[tx] > s_dist[tx + 512]) {
            s_dist[tx] = s_dist[tx + 512];
            s_idx[tx] = s_idx[tx + 512];
        }
        __syncthreads();
    }
    if(tx < 256) {
        if(s_dist[tx] > s_dist[tx + 256]) {
            s_dist[tx] = s_dist[tx + 256];
            s_idx[tx] = s_idx[tx + 256];
        }
        __syncthreads();
    }
    if(tx < 128) {
        if(s_dist[tx] > s_dist[tx + 128]) {
            s_dist[tx] = s_dist[tx + 128];
            s_idx[tx] = s_idx[tx + 128];
        }
        __syncthreads();
    }
    if(tx < 64) {
        if(s_dist[tx] > s_dist[tx + 64]) {
            s_dist[tx] = s_dist[tx + 64];
            s_idx[tx] = s_idx[tx + 64];
        }
        __syncthreads();
    }

    if(tx < 32) warpReduce(s_dist, s_idx, tx);

    if(tx == 0){
        dvc_buffer_dist[blockIdx.x] = s_dist[0];
        if(f_cmp(s_dist[0], __FLT_MAX__) >= 0) {
            dvc_buffer_idx[blockIdx.x] = -1;
        }
        else {
            dvc_buffer_idx[blockIdx.x] = s_idx[0];
        }
    }
}

/*
step 0: 23.192us / 1.05544s 
step 1: 16.807us / 764.85ms 
step 2: 15.700us / 714.49ms 
step 3: 11.821us / 537.97ms
step 4: 11.181us / 508.83ms
1.29220s     45510  28.393us  4.0630us  56.511us  getMin
*/
