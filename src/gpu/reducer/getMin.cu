#include <gpu/reducer.h>
#include <gpu/helper.h>

__global__ void getMin(float *buffer_dist, int *buffer_idx, int *rays_N, int *buffer_N, int *res_idx) {
    const int TS = 1024;

    __shared__ float s_dist[TS];
    __shared__ int s_index[TS];

    int ray_id = blockIdx.x;

    s_index[threadIdx.x] = -1;

    if(threadIdx.x == 0) {
        res_idx[ray_id] = -1;
    }

    int N = buffer_N[0];
    int tx = threadIdx.x;

    __syncthreads();

    if(TS < N) {
        int parts = (N + TS - 1) / TS;

        int initial = parts * threadIdx.x;
        for(int i=initial; i<N && i<(initial + parts); i++) {
            if(s_index[tx] == -1 || (buffer_idx[ray_id * N + i] != -1 && f_cmp(s_dist[tx], buffer_dist[ray_id * N + i]) == 1)) {
                s_index[tx] = buffer_idx[ray_id * N + i];
                s_dist[tx] = buffer_dist[ray_id * N + i];
            }
        }
    }

    __syncthreads();

    int ts = TS;

    while(ts > 1) {
        if(tx >= ts/2) return;

        if(s_index[tx] == -1 || (s_index[tx + ts/2] != -1 && f_cmp(s_dist[tx], s_dist[tx + ts/2]) == 1)) {
            s_index[tx] = s_index[tx + ts/2];
            s_dist[tx] = s_dist[tx + ts/2];
        }

        ts /= 2;
        __syncthreads();
    }

    res_idx[ray_id] = s_index[0];
}
