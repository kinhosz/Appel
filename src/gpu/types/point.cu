#include <gpu/types/point.h>

GPoint::GPoint() {}

GPoint::GPoint(const Point &p) {
    this->x = p.x;
    this->y = p.y;
    this->z = p.z;
}

GPoint::GPoint(const Vetor &v) {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
}