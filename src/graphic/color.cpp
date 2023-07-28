#include <graphic/color.h>
#include <assert.h>

Color::Color(): red(0), green(0), blue(0) {}

Color::Color(int red, int green, int blue) {
    assert(red >= 0 && red <= 255);
    assert(green >= 0 && green <= 255);
    assert(blue >= 0 && blue <= 255);

    this->red = red;
    this->green = green;
    this->blue = blue;
}

int Color::getRed() const {
    return red;
}

int Color::getGreen() const {
    return green;
}

int Color::getBlue() const {
    return blue;
}

double Color::getNormRed() const {
    return (double) red / 255.0;
}

double Color::getNormGreen() const {
    return (double) green / 255.0;
}

double Color::getNormBlue() const {
    return (double) blue / 255.0;
}

void Color::setRed(int red) {
    assert(red >= 0 && red <= 255);
    this->red = red;
}

void Color::setGreen(int green) {
    assert(green >= 0 && green <= 255);
    this->green = green;
}

void Color::setBlue(int blue) {
    assert(blue >= 0 && blue <= 255);
    this->blue = blue;
}