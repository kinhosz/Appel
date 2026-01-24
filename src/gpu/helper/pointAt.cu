#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ GPoint pointAt(GRay ray, float t) {
        GPoint p;
        p.x = ray.location.x + ray.direction.x * t;
        p.y = ray.location.y + ray.direction.y * t;
        p.z = ray.location.z + ray.direction.z * t;

        return p;
    }
}
