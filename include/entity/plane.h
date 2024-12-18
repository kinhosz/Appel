#ifndef ENTITY_PLANE_H
#define ENTITY_PLANE_H

#include <entity/box.h>
#include <geometry/point.h>
#include <geometry/vetor.h>
#include <geometry/ray.h>
#include <graphic/color.h>

class Plane : public Box {
private:
    Point point;
    Vetor normalVector;
    Color color;
public:
    Plane();
    Plane(Point point, Vetor normalVector, Color color);

    Point getPoint() const;
    Vetor getNormalVector() const;
    Color getColor() const;

    void setPoint(Point point);
    void setNormalVector(Vetor normalVector);
    void setColor(Color color);

    SurfaceIntersection intersect(const Ray &ray) const override;
};

#endif
