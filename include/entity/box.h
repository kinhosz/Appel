#ifndef ENTITY_BOX_H
#define ENTITY_BOX_H

#include <graphic/color.h>
#include <geometry/surfaceIntersection.h>
#include <geometry/ray.h>

class Box {
private:
    double diffuseCoefficient;
    double specularCoefficient;
    double ambientCoefficient;
    double reflectionCoefficient;
    double transmissionCoefficient;
    double roughnessCoefficient;
public:
    Box();
    Box(double kd, double ks, double ka, double kr, double kt, double roughness);
    virtual SurfaceIntersection intersect(Ray ray) const;
};

#endif