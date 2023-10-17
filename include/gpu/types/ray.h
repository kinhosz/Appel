#ifndef GRAY_TYPES_GPU_H
#define GRAY_TYPES_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <gpu/types/point.h>
#include <geometry/ray.h>
#include <cuda_runtime.h>

struct GRay {
    GPoint location;
    GPoint direction;

    __host__ __device__ GRay();
    GRay(const Ray &ray); 
};

#endif
#endif
