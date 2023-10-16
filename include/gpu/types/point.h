#ifndef GPOINT_TYPES_GPU_H
#define GPOINT_TYPES_GPU_H

#include <geometry/point.h>
#include <geometry/vetor.h>

struct GPoint {
    float x, y, z;

    GPoint();
    GPoint(const Point &p);
    GPoint(const Vetor &v);
};

#endif
