#ifndef ENTITY_SPHERE_H
#define ENTITY_SPHERE_H

#include <geometry/box.h>
#include <geometry/point.h>
#include <graphic/color.h>

class Sphere : public Box {
private:
    Point center;
    double radius;
    Color color;
public:
    Sphere();
    Sphere(Point center, double radius, Color color, double kd, double ks, double ka, double kr, double kt, double roughness);

    Point getCenter() const;
    double getRadius() const;
    Color getColor() const;

    void setCenter(Point center);
    void setRadius(double radius);
    void setColor(Color color);
};

#endif