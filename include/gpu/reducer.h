#ifndef REDUCER_GPU_H
#define REDUCER_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <cuda_runtime.h>
#include <gpu/types/triangleArray.h>

__global__ void getMin(GTriangleArray *cache, float *dvc_res_dist, int *dvc_buffer_idx, float *dvc_buffer_dist, int *N);

#endif
#endif
