#include <graphic/image.h>
#include <graphic/pixel.h>
#include <geometry/size.h>
#include <SFML/Graphics.hpp>
#include <cassert>

Image::Image() {}

bool Image::loadImage(const std::string& filePath) {
    return image.loadFromFile(filePath);
}

int Image::getWidth() const {
    return int(image.getSize().x);
}

int Image::getHeight() const {
    return int(image.getSize().y);
}

Size Image::getSize() const {
    return Size(int(image.getSize().x), int(image.getSize().y));
}

Pixel Image::getPixel(int x, int y) const {
    assert(x >= 0 && x < int(image.getSize().x));
    assert(y >= 0 && y < int(image.getSize().y));

    sf::Color color = image.getPixel(x, y);
    return Pixel(color.r, color.g, color.b);
}
