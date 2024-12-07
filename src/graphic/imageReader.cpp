#include <graphic/imageReader.h>
#include <graphic/pixel.h>
#include <SFML/Graphics.hpp>

ImageReader::ImageReader() {}

Pixel ImageReader::getPixel(sf::Image image, int x, int y) {
    sf::Color color = image.getPixel(x, y);
    return Pixel(color.r, color.g, color.b);
}
