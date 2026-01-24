#ifndef GEOMETRY_SURFACEINTERSECTION_H
#define GEOMETRY_SURFACEINTERSECTION_H

#include <Appel/graphic/color.h>
#include <Appel/geometry/vetor.h>

namespace Appel {
    struct SurfaceIntersection {
        Color color;
        double distance;
        Vetor normal;

        SurfaceIntersection();
        SurfaceIntersection(Color color, double distance, Vetor normal);

        Vetor getReflection(const Vetor &direction) const;
        Vetor getRefraction(Vetor direction, double refractionIndex) const;
    };
}

#endif
