#include <gpu/helper.h>

__device__ GPoint add(GPoint a, GPoint b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}
