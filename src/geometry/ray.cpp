#include <geometry/ray.h>

Ray::Ray() {
    location = Point();
    direction = Vetor().normalize();   
}

Ray::Ray(Point location, Vetor direction) {
    this->location = location;
    this->direction = direction.normalize();
}

Point Ray::pointAt(double t) const {
    return (direction * t + location).getPoint();
}
