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
    return representation == RGB ? rgb.red : std::runtime_error("You should use getNormRed(), this color is normalized");
}

int Color::getGreen() const {
    return representation == RGB ? rgb.green : std::runtime_error("You should use getNormGreen(), this color is normalized");
}

int Color::getBlue() const {
    return representation == RGB ? rgb.blue : std::runtime_error("You should use getNormBlue(), this color is normalized");
}

double Color::getNormRed() const {
    return representation == NORMALIZED_RGB ? normalized_rgb.red : std::runtime_error("You should use getRed(), this color is not normalized");
}

double Color::getNormGreen() const {
    return representation == NORMALIZED_RGB ? normalized_rgb.green : std::runtime_error("You should use getGreen(), this color is not normalized");
}

double Color::getNormBlue() const {
    return representation == NORMALIZED_RGB ? normalized_rgb.blue : std::runtime_error("You should use getBlue(), this color is not normalized");
}

void Color::setRed(int red) {
    if (representation == NORMALIZED_RGB)
        throw std::runtime_error("You should use setNormRed(), this color is normalized");

    assert(red >= 0 && red <= 255);
    representation = RGB;
    rgb.red = red;
}

void Color::setGreen(int green) {
    if (representation == NORMALIZED_RGB)
        throw std::runtime_error("You should use setNormGreen(), this color is normalized");

    assert(green >= 0 && green <= 255);
    representation = RGB;
    rgb.green = green;
}

void Color::setBlue(int blue) {
    if (representation == NORMALIZED_RGB)
        throw std::runtime_error("You should use setNormBlue(), this color is normalized");

    assert(blue >= 0 && blue <= 255);
    representation = RGB;
    rgb.blue = blue;
}

void Color::setNormRed(double normRed) {
    if (representation == RGB)
        throw std::runtime_error("You should use setRed(), this color is not normalized");

    assert(normRed >= 0.0 && normRed <= 1.0);
    representation = NORMALIZED_RGB;
    normalized_rgb.red = normRed;
}

void Color::setNormGreen(double normGreen) {
    if (representation == RGB)
        throw std::runtime_error("You should use setGreen(), this color is not normalized");

    assert(normGreen >= 0.0 && normGreen <= 1.0);
    representation = NORMALIZED_RGB;
    normalized_rgb.green = normGreen;
}

void Color::setNormBlue(double normBlue) {
    if (representation == RGB)
        throw std::runtime_error("You should use setBlue(), this color is not normalized");

    assert(normBlue >= 0.0 && normBlue <= 1.0);
    representation = NORMALIZED_RGB;
    normalized_rgb.blue = normBlue;
}