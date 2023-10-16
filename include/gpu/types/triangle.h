#ifndef GTRIANGLE_TYPES_GPU_H
#define GTRIANGLE_TYPES_GPU_H

#include <gpu/types/point.h>

struct GTriangle {
    GPoint point[3];
    int host_id;

    GTriangle();
    GTriangle(const Triangle &t, int host_id);
};

#endif
