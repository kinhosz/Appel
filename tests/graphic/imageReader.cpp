#include <graphic/imageReader.h>
#include <graphic/pixel.h>
#include <assert.h>

int main() {
    ImageReader imageReader;
    Pixel p1, p2;

    if (!imageReader.loadImage("assets/outputs/view/humanFace.png")) {
        return -1;
    }

    p1 = imageReader.getPixel(0, 0);
    p2 = imageReader.getPixel(320, 180);

    assert(p1.getRed() == 0);
    assert(p1.getGreen() == 0);
    assert(p1.getBlue() == 0);

    assert(p2.getRed() == 154);
    assert(p2.getGreen() == 154);
    assert(p2.getBlue() == 154);
}
