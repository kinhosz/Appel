#ifndef TRIANGLE_TYPES_GPU_H
#define TRIANGLE_TYPES_GPU_H

struct GTriangle {
    float p0x, p0y, p0z;
    float p1x, p1y, p1z;
    float p2x, p2y, p2z;
    int host_id;
};

#endif
