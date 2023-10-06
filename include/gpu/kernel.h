#ifndef KERNEL_GPU_H
#define KERNEL_GPU_H

#include <gpu/types/ray.h>
#include <gpu/types/triangle.h>

__global__ void castRay(GTriangle *cache, GRay *ray, float *block_dist, int *block_idx, int N);

#endif
