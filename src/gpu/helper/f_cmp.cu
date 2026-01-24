#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ int f_cmp(float a, float b) {
        float eps = 0.0005;

        if(fabs(a - b) < eps) return 0;
        if(a < b) return -1;
        return 1;
    }
}
