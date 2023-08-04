#include <graphic/color.h>
#include <assert.h>

Color::Color() {
    red = 0.0;
    green = 0.0;
    blue = 0.0;
}

Color::Color(double red, double green, double blue) {
    assert(red >= 0.0 && red <= 1.0);
    assert(green >= 0.0 && green <= 1.0);
    assert(blue >= 0.0 && blue <= 1.0);

    this->red = red;
    this->green = green;
    this->blue = blue;
}

double Color::getRed() const {
    return red;
}

double Color::getGreen() const {
    return green;
}

double Color::getBlue() const {
    return blue;
}

void Color::setRed(double red) {
    assert(red >= 0.0 && red <= 1.0);

    this->red = red;
}

void Color::setGreen(double green) {
    assert(green >= 0.0 && green <= 1.0);

    this->green = green;
}

void Color::setBlue(double blue) {
    assert(blue >= 0.0 && blue <= 1.0);

    this->blue = blue;
}