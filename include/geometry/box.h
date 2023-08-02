#ifndef GEOMETRY_BOX_H
#define GEOMETRY_BOX_H

#include <graphic/color.h>

class Box {
private:
    double diffuse_coefficient;
    double specular_coefficient;
    double ambient_coefficient;
    double reflection_coefficient;
    double transmission_coefficient;
    double roughness;
public:
    Box();
    Box(double kd, double ks, double ka, double kr, double kt, double roughness);
};

#endif