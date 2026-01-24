#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ GPoint sub(GPoint a, GPoint b) {
        a.x -= b.x;
        a.y -= b.y;
        a.z -= b.z;

        return a;
    }
}
