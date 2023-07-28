#include <light.h>
#include <assert.h>

Light::Light(Point location, int red, int green, int blue) {
    assert(red >= 0 && red <= 255);
    assert(green >= 0 && green <= 255);
    assert(blue >= 0 && blue <= 255);

    this->location = location;
    this->red = red;
    this->green = green;
    this->blue = blue;
}

Point Light::getLocation() const {
    return location;
}

int Light::getRed() const {
    return red;
}

int Light::getGreen() const {
    return green;
}

int Light::getBlue() const {
    return blue;
}

void Light::setLocation(Point location) {
    this->location = location;
}

void Light::setRed(int red) {
    assert(red >= 0 && red <= 255);
    this->red = red;
}

void Light::setGreen(int green) {
    assert(green >= 0 && green <= 255);
    this->green = green;
}

void Light::setBlue(int blue) {
    assert(blue >= 0 && blue <= 255);
    this->blue = blue;
}