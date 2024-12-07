#include <graphic/image.h>
#include <graphic/pixel.h>
#include <cassert>

int main() {
    Image image;
    Pixel p1, p2;
    int middleX, middleY;

    if (!image.loadImage("assets/outputs/view/humanFace.png")) {
        return -1;
    }

    middleX = image.getWidth()/2;
    middleY = image.getHeight()/2;

    assert(middleX == 320);
    assert(middleY == 180);

    p1 = image.getPixel(0, 0);
    p2 = image.getPixel(middleX, middleY);

    assert(p1.getRed() == 0);
    assert(p1.getGreen() == 0);
    assert(p1.getBlue() == 0);

    assert(p2.getRed() == 154);
    assert(p2.getGreen() == 154);
    assert(p2.getBlue() == 154);
}
