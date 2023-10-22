#ifndef GPOINT_TYPES_GPU_H
#define GPOINT_TYPES_GPU_H

#include <geometry/point.h>
#include <geometry/vetor.h>

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif

struct GPoint {
    float x, y, z;

#ifndef APPEL_GPU_DISABLED
    __host__ __device__ GPoint() = default;
#else
    GPoint();
#endif

    GPoint(const Point &p);
    GPoint(const Vetor &v);
};

#endif
