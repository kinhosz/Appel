#ifndef GEOMETRY_BOX_H
#define GEOMETRY_BOX_H

#include <graphic/color.h>

struct Box {
    double kd, ks, ka, kr, kt, roughness;
    Color color;

    Box();
    Box(double kd, double ks, double ka, double kr, double kt, double roughness, Color color);
};

#endif