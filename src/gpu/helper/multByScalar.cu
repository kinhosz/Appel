#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ GPoint multByScalar(GPoint p, float s) {
        p.x *= s;
        p.y *= s;
        p.z *= s;

        return p;
    }
}
