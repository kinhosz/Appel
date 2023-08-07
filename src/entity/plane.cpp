#include <entity/plane.h>

Plane::Plane() : Box(), point(Point()), normalVector(Vetor()), color(Color()) {}

Plane::Plane(Point point, Vetor normalVector, Color color, double kd, double ks, double ka, double kr, double kt, double roughness)
    : Box(kd, ks, ka, kr, kt, roughness), point(point), normalVector(normalVector), color(color) {
}

Point Plane::getPoint() const {
    return this->point;
}

Vetor Plane::getNormalVector() const {
    return this->normalVector;
}

Color Plane::getColor() const {
    return this->color;
}

void Plane::setPoint(Point point) {
    this->point = point;
}

void Plane::setNormalVector(Vetor normalVector) {
    this->normalVector = normalVector;
}

void Plane::setColor(Color color) {
    this->color = color;
}