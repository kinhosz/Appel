#ifndef ENTITY_LIGHT_H
#define ENTITY_LIGHT_H

#include <geometry/point.h>
#include <graphic/color.h>

class Light {
private:
    Point location;
    Color intensity;
public:
    Light();
    Light(Point location, Color intensity);

    Point getLocation() const;
    Color getIntensity() const;

    void setLocation(Point location);
    void setIntensity(Color intensity);

    bool operator==(const Light& other) const;
    bool operator!=(const Light& other) const;
};

#endif