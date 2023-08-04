#ifndef ENTITY_LIGHT_H
#define ENTITY_LIGHT_H

#include <geometry/point.h>
#include <graphic/pixel.h>

class Light {
private:
    Point location;
    Pixel intensity;
public:
    Light(Point location, Pixel intensity);

    Point getLocation() const;
    Pixel getIntensity() const;

    void setLocation(Point location);
    void setIntensity(Pixel intensity);
};

#endif