#ifndef GTRIANGLEARRAY_TYPES_GPU_H
#define GTRIANGLEARRAY_TYPES_GPU_H

#include <gpu/types/pointArray.h>

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif

struct GTriangleArray {
    GPointArray point[3];
    int *host_id;
};

#endif
