#ifndef KERNEL_GPU_H
#define KERNEL_GPU_H

#include <cuda_runtime.h>
#include <gpu/types/triangle.h>
#include <gpu/types/ray.h>

__global__ void updateCacheTriangle(int device_id, GTriangle triangle, GTriangle* cache);

__global__ void updateCacheLight(int device_id, GPoint p, GPoint* light);

__global__ void traceRayPreProcess(const GPoint up, const GPoint right, const GPoint front,
    const GPoint loc, const float dist, const int height, const int width,
    GTriangle* cache_triangle, GPoint* cache_light, const int depth, const int lights,
    int* buffer);

#endif
