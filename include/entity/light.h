#ifndef ENTITY_LIGHT_H
#define ENTITY_LIGHT_H

#include "point.h"

struct Light {
    Point location;

    struct Intensity {
        int red;
        int green;
        int blue;

    private:
        int clamp(int value);
    public:
        void setRed(int r);
        void setGreen(int g);
        void setBlue(int b);
        
    } intensity;

    Light();
    Light(Point location, Intensity intensity);
};

#endif
