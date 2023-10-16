#include <gpu/helper.h>

__device__ float norm(GPoint a) {
    float d = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);

    return d;
}
