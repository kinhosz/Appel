#include <Appel/gpu/helper.h>

__device__ GPoint multByScalar(GPoint p, float s) {
    p.x *= s;
    p.y *= s;
    p.z *= s;

    return p;
}
