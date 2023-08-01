#include <graphic/color.h>
#include <assert.h>

Color::Color() : representation(RGB), rgb{0, 0, 0} {}

Color::Color(int red, int green, int blue, ColorRepresentation representation)
    : representation(representation) {
    assert(red >= 0 && red <= 255);
    assert(green >= 0 && green <= 255);
    assert(blue >= 0 && blue <= 255);

    rgb.red = red;
    rgb.green = green;
    rgb.blue = blue;
}

Color::Color(double normRed, double normGreen, double normBlue, ColorRepresentation representation)
    : representation(representation) {
    assert(normRed >= 0.0 && normRed <= 1.0);
    assert(normGreen >= 0.0 && normGreen <= 1.0);
    assert(normBlue >= 0.0 && normBlue <= 1.0);

    normalized_rgb.red = normRed;
    normalized_rgb.green = normGreen;
    normalized_rgb.blue = normBlue;
}

int Color::getRed() const {
    assert(representation == RGB);
    return rgb.red;
}

int Color::getGreen() const {
    assert(representation == RGB);
    return rgb.green;
}

int Color::getBlue() const {
    assert(representation == RGB);
    return rgb.blue;
}

double Color::getNormRed() const {
    assert(representation == NORMALIZED_RGB);
    return normalized_rgb.red;
}

double Color::getNormGreen() const {
    assert(representation == NORMALIZED_RGB);
    return normalized_rgb.green;
}

double Color::getNormBlue() const {
    assert(representation == NORMALIZED_RGB);
    return normalized_rgb.blue;
}

void Color::setRed(int red) {
    assert(representation == RGB);
    assert(red >= 0 && red <= 255);
    representation = RGB;
    rgb.red = red;
}

void Color::setGreen(int green) {
    assert(representation == RGB);
    assert(green >= 0 && green <= 255);
    representation = RGB;
    rgb.green = green;
}

void Color::setBlue(int blue) {
    assert(representation == RGB);
    assert(blue >= 0 && blue <= 255);
    representation = RGB;
    rgb.blue = blue;
}

void Color::setNormRed(double normRed) {
    assert(representation == NORMALIZED_RGB);
    assert(normRed >= 0.0 && normRed <= 1.0);
    representation = NORMALIZED_RGB;
    normalized_rgb.red = normRed;
}

void Color::setNormGreen(double normGreen) {
    assert(representation == NORMALIZED_RGB);
    assert(normGreen >= 0.0 && normGreen <= 1.0);
    representation = NORMALIZED_RGB;
    normalized_rgb.green = normGreen;
}

void Color::setNormBlue(double normBlue) {
    assert(representation == NORMALIZED_RGB);
    assert(normBlue >= 0.0 && normBlue <= 1.0);
    representation = NORMALIZED_RGB;
    normalized_rgb.blue = normBlue;
}