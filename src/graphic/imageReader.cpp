#include <graphic/imageReader.h>
#include <graphic/pixel.h>
#include <SFML/Graphics.hpp>
#include <assert.h>

ImageReader::ImageReader() {}

bool ImageReader::loadImage(const std::string& filePath) {
    return image.loadFromFile(filePath);
}

Pixel ImageReader::getPixel(int x, int y) const {
    assert(x >= 0 && x < static_cast<int>(image.getSize().x));
    assert(y >= 0 && y < static_cast<int>(image.getSize().y));

    sf::Color color = image.getPixel(x, y);
    return Pixel(color.r, color.g, color.b);
}
