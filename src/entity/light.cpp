#include <light.h>
#include <cassert>

Light::Light(Point location, int red, int green, int blue) : location(location), red(red), green(green), blue(blue) {}

Light Light::create(Point location, int red, int green, int blue) {
    assert(red >= 0 && red <= 255);
    assert(green >= 0 && green <= 255);
    assert(blue >= 0 && blue <= 255);

    return Light(location, red, green, blue);
}

Point Light::get_location() const {
    return location;
}

int Light::get_red() const {
    return red;
}

int Light::get_green() const {
    return green;
}

int Light::get_blue() const {
    return blue;
}

int Light::get_rgb() const {
    return (red << 16) + (green << 8) + blue;
}

void Light::set_location(Point location) {
    this->location = location;
}

void Light::set_red(int red) {
    assert(red >= 0 && red <= 255);

    this->red = red;
}

void Light::set_green(int green) {
    assert(green >= 0 && green <= 255);

    this->green = green;
}

void Light::set_blue(int blue) {
    assert(blue >= 0 && blue <= 255);

    this->blue = blue;
}

void Light::set_rgb(int red, int green, int blue) {
    assert(red >= 0 && red <= 255);
    assert(green >= 0 && green <= 255);
    assert(blue >= 0 && blue <= 255);

    this->red = red;
    this->green = green;
    this->blue = blue;
}