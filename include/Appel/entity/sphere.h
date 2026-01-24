#ifndef ENTITY_SPHERE_H
#define ENTITY_SPHERE_H

#include <Appel/entity/box.h>
#include <Appel/geometry/point.h>
#include <Appel/geometry/ray.h>
#include <Appel/graphic/color.h>

namespace Appel {
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
}

#endif
