#include <light.h>

Light::Light() : location(Point()), intensity(Intensity()) {}

Light::Light(Point location, Intensity intensity) : location(location), intensity(intensity) {}

int Light::Intensity::clamp(int value) {
    if (value < 0)
        return 0;
    else if (value > 255)
        return 255;
    else
        return value;
}

void Light::Intensity::setRed(int r) {
    red = clamp(r);
}

void Light::Intensity::setGreen(int g) {
    green = clamp(g);
}

void Light::Intensity::setBlue(int b) {
    blue = clamp(b);
}