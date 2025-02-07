#ifndef ENTITY_SPHERE_H
#define ENTITY_SPHERE_H

#include <entity/box.h>
#include <geometry/point.h>
#include <geometry/ray.h>
#include <graphic/color.h>

class Sphere : public Box {
private:
    Point center;
    double radius;
    Color color;
public:
    Sphere();
    Sphere(Point center, double radius, Color color);

    Point getCenter() const;
    double getRadius() const;
    Color getColor() const;

    void setCenter(Point center);
    void setRadius(double radius);
    void setColor(Color color);

    SurfaceIntersection intersect(const Ray &ray) const override;
};

#endif
