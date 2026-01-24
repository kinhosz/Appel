#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ GPoint cross(GPoint a, GPoint b) {
        float i = (a.y * b.z) - (a.z * b.y);
        float j = (a.z * b.x) - (a.x * b.z);
        float k = (a.x * b.y) - (a.y * b.x);

        GPoint p;
        p.x = i;
        p.y = j;
        p.z = k;

        return p;
    }
}
