#ifndef KERNEL_GPU_H
#define KERNEL_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <cuda_runtime.h>
#include <gpu/types/triangle.h>
#include <gpu/types/triangleArray.h>
#include <gpu/types/ray.h>

__global__ void updateCache(int device_id, GTriangle triangle, GTriangle* cache);

__global__ void castRay(GRay *rays, int *rays_N, GTriangleArray *cache, int *triangles_N, float *res_dist, int *table);

#endif
#endif
