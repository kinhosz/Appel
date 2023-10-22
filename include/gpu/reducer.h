#ifndef REDUCER_GPU_H
#define REDUCER_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <cuda_runtime.h>

__global__ void getMin(float *buffer_dist, int *buffer_idx, 
    int *rays_N, int *buffer_N, int *res_idx);

#endif
#endif
