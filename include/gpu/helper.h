#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#ifndef APPEL_GPU_DISABLED

#include <cuda_runtime.h>
#include <gpu/types/point.h>
#include <gpu/types/ray.h>
#include <gpu/types/triangle.h>

/* ray tracing */
__device__ GRay createRay(const GPoint up, const GPoint right, 
    const GPoint front, const GPoint loc, const float dist,
    const int height, const int width, const int i);

__device__ int castRay(GRay ray, GTriangle* cache_triangle);

__device__ void brightnessCastRay(int pos, GTriangle surface, int lightId,
    GRay ray, GTriangle* cache_triangle, GPoint* cache_light, int* buffer);

/* vector operations */
__device__ GPoint multByScalar(GPoint p, float s);

__device__ GPoint normalize(GPoint p);

__device__ GPoint pointAt(GRay ray, float t);

__device__ float angle(GPoint a, GPoint b);

__device__ GPoint sub(GPoint a, GPoint b);

__device__ GPoint add(GPoint a, GPoint b);

__device__ GPoint cross(GPoint a, GPoint b);

__device__ float dot(GPoint a, GPoint b);

__device__ float norm(GPoint a);

/* surface operations */
__device__ GPoint getReflection(GTriangle triangle, GPoint dir);

__device__ GPoint getRefraction(GTriangle surface, GPoint dir, float refIndex);

__device__ GPoint getNormal(GTriangle triangle);

__device__ float getDistance(GRay ray, GTriangle triangle);

__device__ int isInside(GPoint point, GTriangle triangle);

/* float operations */
__device__ int f_cmp(float a, float b);

#endif
#endif
