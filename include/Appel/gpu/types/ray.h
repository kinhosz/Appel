#ifndef GRAY_TYPES_GPU_H
#define GRAY_TYPES_GPU_H

#include <Appel/gpu/types/point.h>
#include <Appel/geometry/ray.h>

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif 

struct GRay {
    GPoint location;
    GPoint direction;

#ifndef APPEL_GPU_DISABLED
    __host__ __device__ GRay();
#else
    GRay();
#endif

    GRay(const Ray &ray); 
};

#endif
