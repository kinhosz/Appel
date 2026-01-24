#ifndef GTRIANGLE_TYPES_GPU_H
#define GTRIANGLE_TYPES_GPU_H

#include <Appel/gpu/types/point.h>
#include <Appel/geometry/triangle.h>

#ifndef APPEL_GPU_DISABLED
#include <cuda_runtime.h>
#endif

namespace Appel {
    struct GTriangle {
        GPoint point[3];
        int host_id;

    #ifndef APPEL_GPU_DISABLED
        __host__ __device__ GTriangle();
    #else
        GTriangle();
    #endif

        GTriangle(const Triangle &t, int host_id);
    };
}
#endif
