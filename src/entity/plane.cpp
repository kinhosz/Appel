#include <entity/plane.h>
#include <geometry/utils.h>

Plane::Plane() : Box(), point(Point()), normalVector(Vetor()), color(Color()) {}

Plane::Plane(Point point, Vetor normalVector, Color color, double kd, double ks, double ka, double kr, double kt, double roughness)
    : Box(kd, ks, ka, kr, kt, roughness), point(point), normalVector(normalVector), color(color)
{
}

Point Plane::getPoint() const
{
    return this->point;
}

Vetor Plane::getNormalVector() const
{
    return this->normalVector;
}

Color Plane::getColor() const
{
    return this->color;
}

void Plane::setPoint(Point point)
{
    this->point = point;
}

void Plane::setNormalVector(Vetor normalVector)
{
    this->normalVector = normalVector;
}

void Plane::setColor(Color color)
{
    this->color = color;
}

SurfaceIntersection Plane::intersect(const Ray &ray) const
{
    Vetor normal = normalVector.normalize();

    if (normal.isOrthogonal(ray.direction))
        return SurfaceIntersection();

    double D = -normal.x * this->point.x - normal.y * this->point.y - normal.z * this->point.z;
    double A = normal.x, B = normal.y, C = normal.z;

    double c1 = (A * ray.location.x + B * ray.location.y + C * ray.location.z + D);
    double c2 = (A * ray.direction.x + B * ray.direction.y + C * ray.direction.z);

    double t = -c1 / c2;

    if (cmp(t, 0) == -1)
        return SurfaceIntersection();

    return SurfaceIntersection(color, t, normal);
}
