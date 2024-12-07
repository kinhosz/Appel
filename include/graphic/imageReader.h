#ifndef GRAPHIC_IMAGE_READER_H
#define GRAPHIC_IMAGE_READER_H

#include <graphic/pixel.h>
#include <SFML/Graphics.hpp>
#include <string>

class ImageReader {
private:
    sf::Image image;
public:
    ImageReader();
    bool loadImage(const std::string& filePath);
    Pixel getPixel(int x, int y) const;
};

#endif
