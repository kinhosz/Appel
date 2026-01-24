#include <Appel/graphic/pixel.h>
#include <assert.h>

Pixel::Pixel() {
    red = 0;
    green = 0;
    blue = 0;
}

Pixel::Pixel(Color color) {
    red = color.getRed() * 255;
    green = color.getGreen() * 255;
    blue = color.getBlue() * 255;
}

Pixel::Pixel(int red, int green, int blue) {
    assert(red >= 0 && red <= 255);
    assert(green >= 0 && green <= 255);
    assert(blue >= 0 && blue <= 255);

    this->red = red;
    this->green = green;
    this->blue = blue;
}

int Pixel::getRed() const {
    return red;
}

int Pixel::getGreen() const {
    return green;
}

int Pixel::getBlue() const {
    return blue;
}

void Pixel::setRed(int red) {
    assert(red >= 0 && red <= 255);

    this->red = red;
}

void Pixel::setGreen(int green) {
    assert(green >= 0 && green <= 255);

    this->green = green;
}

void Pixel::setBlue(int blue) {
    assert(blue >= 0 && blue <= 255);

    this->blue = blue;
}
