#ifndef GEOMETRY_SURFACEINTERSECTION_H
#define GEOMETRY_SURFACEINTERSECTION_H

#include <graphic/color.h>
#include <geometry/vetor.h>

struct SurfaceIntersection {
    Color color;
    double distance;
    Vetor normal;

    SurfaceIntersection();
    SurfaceIntersection(Color color, double distance, Vetor normal);
};

#endif
