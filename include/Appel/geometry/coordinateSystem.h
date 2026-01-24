#ifndef GEOMETRY_COORDINATE_SYSTEM_H
#define GEOMETRY_COORDINATE_SYSTEM_H

#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>

struct CoordinateSystem {
    Point origin;
    double angle_x, angle_y, angle_z;

    CoordinateSystem();
    CoordinateSystem(Point origin, Vetor ux, Vetor uy, Vetor uz);

    Point rebase(const Point& p) const;
};

#endif
