#include <graphic/imageReader.h>
#include <graphic/pixel.h>
#include <SFML/Graphics.hpp>
#include <assert.h>

int main(){
    sf::Image image;
    ImageReader imageReader;
    Pixel p1, p2;

    image.loadFromFile("assets/outputs/view/humanFace.png");

    p1 = imageReader.getPixel(image, 0, 0);
    p2 = imageReader.getPixel(image, 320, 180); // Get the pixel in the middle of the image

    assert(p1.getRed() == 0);
    assert(p1.getGreen() == 0);
    assert(p1.getBlue() == 0);

    assert(p2.getRed() == 154);
    assert(p2.getGreen() == 154);
    assert(p2.getBlue() == 154);
}
