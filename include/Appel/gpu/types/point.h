#ifndef GPOINT_TYPES_GPU_H
#define GPOINT_TYPES_GPU_H

#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif

struct GPoint {
    float x, y, z;

#ifndef APPEL_GPU_DISABLED
    __host__ __device__ GPoint();
#else
    GPoint();
#endif

    GPoint(const Point &p);
    GPoint(const Vetor &v);
};

#endif
