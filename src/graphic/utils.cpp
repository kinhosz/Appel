#include <graphic/utils.h>
#include <SFML/Graphics.hpp>
#include <iostream>

bool saveAsPng(const Frame& frame, const std::string& filename) {
    sf::Image image;

    image.create(frame.horizontal(), frame.vertical());

    for(int i=0;i<frame.horizontal();i++){
        for(int j=0;j<frame.vertical();j++){
            Pixel pixel = frame.getPixel(i, j);

            image.setPixel(i, j, sf::Color(pixel.red, pixel.green, pixel.blue));
        }
    }

    return image.saveToFile(filename);
}
