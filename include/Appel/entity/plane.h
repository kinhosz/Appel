#ifndef ENTITY_PLANE_H
#define ENTITY_PLANE_H

#include <Appel/entity/box.h>
#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>
#include <Appel/geometry/ray.h>
#include <Appel/graphic/color.h>

namespace Appel {
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
}

#endif
