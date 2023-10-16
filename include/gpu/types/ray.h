#ifndef GRAY_TYPES_GPU_H
#define GRAY_TYPES_GPU_H

#include <gpu/types/point.h>
#include <geometry/ray.h>

struct GRay {
    GPoint location;
    GPoint direction;

    GRay();
    GRay(const Ray &ray); 
};

#endif
