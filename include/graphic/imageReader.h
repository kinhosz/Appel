#ifndef GRAPHIC_IMAGE_READER_H
#define GRAPHIC_IMAGE_READER_H

#include <graphic/pixel.h>
#include <SFML/Graphics.hpp>

class ImageReader {
public:
    ImageReader();
    Pixel getPixel(sf::Image image, int x, int y);
};

#endif
