#ifndef KERNEL_GPU_H
#define KERNEL_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <cuda_runtime.h>
#include <Appel/gpu/types/triangle.h>
#include <Appel/gpu/types/ray.h>

namespace Appel {
  __global__ void updateCache(int device_id, GTriangle triangle, GTriangle* cache);

  __global__ void castRay(GRay* ray, float* buffer_dist, int* buffer_idx, GTriangle* cache, int* N);
}

#endif
#endif
