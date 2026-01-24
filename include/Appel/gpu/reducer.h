#ifndef REDUCER_GPU_H
#define REDUCER_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <cuda_runtime.h>

namespace Appel {
  __global__ void getMin(float* buffer_dist, int* buffer_idx, int* N, int* res);
}

#endif
#endif
