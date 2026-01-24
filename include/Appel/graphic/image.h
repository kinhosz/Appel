#ifndef GRAPHIC_IMAGE_H
#define GRAPHIC_IMAGE_H

#include <Appel/graphic/pixel.h>
#include <SFML/Graphics.hpp>
#include <string>

namespace Appel {
    class Image {
    private:
        sf::Image image;
    public:
        Image();
        
        bool loadImage(const std::string& filePath);
        
        int getWidth() const;
        int getHeight() const;

        Pixel getPixel(int x, int y) const;
    };
}

#endif
