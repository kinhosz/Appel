#ifndef GEOMETRY_BOX_H
#define GEOMETRY_BOX_H

#include <graphic/color.h>

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
};

#endif