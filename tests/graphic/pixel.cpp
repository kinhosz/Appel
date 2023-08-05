#include <graphic/pixel.h>
#include <graphic/color.h>
#include <assert.h>
using namespace std;

int main() {
    Pixel p1, p2(255, 174, 10), p3(Color(0.2, 0.8, 0.4));

    assert(p1.getRed() == 0);
    assert(p1.getGreen() == 0);
    assert(p1.getBlue() == 0);

    assert(p2.getRed() == 255);
    assert(p2.getGreen() == 174);
    assert(p2.getBlue() == 10);

    assert(p3.getRed() == 51);
    assert(p3.getGreen() == 204);
    assert(p3.getBlue() == 102);

    p1.setRed(255);
    p1.setGreen(174);
    p1.setBlue(10);

    p2.setRed(15);
    p2.setGreen(20);
    p2.setBlue(215);

    p3.setRed(200);
    p3.setGreen(100);
    p3.setBlue(50);

    assert(p1.getRed() == 255);
    assert(p1.getGreen() == 174);
    assert(p1.getBlue() == 10);

    assert(p2.getRed() == 15);
    assert(p2.getGreen() == 20);
    assert(p2.getBlue() == 215);

    assert(p3.getRed() == 200);
    assert(p3.getGreen() == 100);
    assert(p3.getBlue() == 50);
}
