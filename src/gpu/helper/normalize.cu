#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ GPoint normalize(GPoint p) {
        float r = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        p.x /= r;
        p.y /= r;
        p.z /= r;
        
        return p;
    }
}
