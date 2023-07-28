#ifndef ENTITY_LIGHT_H
#define ENTITY_LIGHT_H

#include "point.h"

class Light {
private:
    Point location;
    int red, green, blue;
    Light(Point location, int red, int green, int blue);
public:
    static Light create(Point location, int red, int green, int blue);
    Point get_location() const;
    int get_red() const;
    int get_green() const;
    int get_blue() const;
    int get_rgb() const;
    void set_location(Point location);
    void set_red(int red);
    void set_green(int green);
    void set_blue(int blue);
    void set_rgb(int red, int green, int blue);
};

#endif
