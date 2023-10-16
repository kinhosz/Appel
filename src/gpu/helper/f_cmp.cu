#include <gpu/helper.h>

__device__ int f_cmp(float a, float b) {
    float eps = 1e-9;

    if(fabs(a - b) < eps) return 0;
    if(a < b) return -1;
    else return 1;
}
