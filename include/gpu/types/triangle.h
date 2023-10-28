#ifndef GTRIANGLE_TYPES_GPU_H
#define GTRIANGLE_TYPES_GPU_H

#include <gpu/types/point.h>
#include <geometry/triangle.h>

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif

struct GTriangle {
    GPoint point[3];
    int host_id;

#ifndef APPEL_GPU_DISABLED
    __host__ __device__ GTriangle() = default;
#else
    GTriangle();
#endif

    GTriangle(const Triangle &t, int host_id);
};
#endif
