#ifndef GTRIANGLE_TYPES_GPU_H
#define GTRIANGLE_TYPES_GPU_H

#include <gpu/types/point.h>
#include <geometry/triangle.h>
#include <cuda_runtime.h>

struct GTriangle {
    GPoint point[3];
    int host_id;

    __host__ __device__ GTriangle();
    GTriangle(const Triangle &t, int host_id);
};

#endif
