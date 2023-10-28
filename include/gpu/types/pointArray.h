#ifndef GPOINTARRAY_TYPES_GPU_H
#define GPOINTARRAY_TYPES_GPU_H

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif

struct GPointArray {
    float *x;
    float *y;
    float *z;
};

#endif
