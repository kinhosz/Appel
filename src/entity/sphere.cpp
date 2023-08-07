#include <entity/sphere.h>

Sphere::Sphere() : Box(), center(Point()), radius(0.0), color(Color()) {}

Sphere::Sphere(Point center, double radius, Color color, double kd, double ks, double ka, double kr, double kt, double roughness)
    : Box(kd, ks, ka, kr, kt, roughness), center(center), radius(radius), color(color) {
}

Point Sphere::getCenter() const {
    return center;
}

double Sphere::getRadius() const {
    return radius;
}

Color Sphere::getColor() const {
    return color;
}

void Sphere::setCenter(Point center) {
    this->center = center;
}

void Sphere::setRadius(double radius) {
    this->radius = radius;
}

void Sphere::setColor(Color color) {
    this->color = color;
}