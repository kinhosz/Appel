#include <Appel/gpu/types/ray.h>

namespace Appel {
    __host__ __device__ GRay::GRay() {}

    GRay::GRay(const Ray &ray) {
        this->location = GPoint(ray.location);
        this->direction = GPoint(ray.direction);
    }
}
