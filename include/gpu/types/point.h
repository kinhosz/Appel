#ifndef GPOINT_TYPES_GPU_H
#define GPOINT_TYPES_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <geometry/point.h>
#include <geometry/vetor.h>
#include <cuda_runtime.h>

struct GPoint {
    float x, y, z;

    __host__ __device__ GPoint();
    GPoint(const Point &p);
    GPoint(const Vetor &v);
};

#endif
#endif
