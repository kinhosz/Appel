#include <entity/sphere.h>
#include <geometry/vetor.h>
#include <geometry/utils.h>
#include <cmath>

Sphere::Sphere() : Box(), center(Point()), radius(0.0), color(Color()) {}

Sphere::Sphere(Point center, double radius, Color color)
    : Box(), center(center), radius(radius), color(color) {}

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

SurfaceIntersection Sphere::intersect(const Ray &ray) const {
    Vetor oc = Vetor(ray.location) - Vetor(this->center);
    double a = ray.direction.dot(ray.direction);
    double b = 2.0 * oc.dot(ray.direction);
    double c = oc.dot(oc) - (this->radius * this->radius);
    double discriminant = (b * b) - (4 * a * c);

    if (cmp(discriminant, 0) == -1)
        return SurfaceIntersection();

    double t = (-b - sqrt(discriminant)) / (2.0 * a);

    if (cmp(t, 0) == -1)
        return SurfaceIntersection();

    Vetor normal = (Vetor(ray.pointAt(t)) - Vetor(this->center)).normalize();

    return SurfaceIntersection(this->color, t, normal);
}