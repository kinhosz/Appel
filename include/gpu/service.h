#ifndef SERVICE_GPU_H
#define SERVICE_GPU_H

#include <gpu/types/ray.h>
#include <gpu/types/triangle.h>

__device__ float planeIntersect(GRay ray, GTriangle gt);
__device__ int onTriangle(GTriangle gt, GRay ray, float t);

#endif
