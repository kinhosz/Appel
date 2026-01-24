#ifndef GEOMETRY_RAY_H
#define GEOMETRY_RAY_H

#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>

struct Ray {
    Point location;
    Vetor direction;

    Ray();
    Ray(Point location, Vetor direction);
    Point pointAt(double t) const;
};

#endif
