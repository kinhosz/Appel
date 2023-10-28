#include <gpu/helper.h>

__device__ int f_cmp(float a, float b) {
    float eps = 0.0005;

    int ret = ((a<b)? -1: 1);
    ret = (fabs(a - b) < eps ? 0: ret);

    return ret;
}
