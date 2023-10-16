#include <gpu/types/ray.h>

GRay::GRay() {}

GRay::GRay(const Ray &ray) {
    this->location = GPoint(ray.location);
    this->direction = GPoint(ray.direction);
}
