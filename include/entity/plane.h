#ifndef ENTITY_PLANE_H
#define ETNITY_PLANE_H

#include <geometry/box.h>
#include <geometry/point.h>
#include <geometry/vetor.h>
#include <graphic/color.h>

class Plane : public Box {
private:
    Point point;
    Vetor normalVector;
    Color color;
public:
    Plane();
    Plane(Point point, Vetor normalVector, Color color, double kd, double ks, double ka, double kr, double kt, double roughness);

    Point getPoint() const;
    Vetor getNormalVector() const;
    Color getColor() const;

    void setPoint(Point point);
    void setNormalVector(Vetor normalVector);
    void setColor(Color color);
};

#endif