#ifndef ENTITY_LIGHT_H
#define ENTITY_LIGHT_H

#include "point.h"

class Light {
private:
    Point location;
    int red, green, blue;
public:
    Light(Point location, int red, int green, int blue);

    Point getLocation() const;
    int getRed() const;
    int getGreen() const;
    int getBlue() const;

    void setLocation(Point location);
    void setRed(int red);
    void setGreen(int green);
    void setBlue(int blue);
};

#endif