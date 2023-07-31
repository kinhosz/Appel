#ifndef GEOMETRY_RAY_H
#define GEOMETRY_RAY_H

#include <geometry/point.h>
#include <geometry/vetor.h>

struct Ray {
    Point location;
    Vetor direction;

    Ray();
    Ray(Point location, Vetor direction);
    Point pointAt(double t) const;
};

#endif
