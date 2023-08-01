#include <graphic/color.h>
#include <assert.h>
#include <cmath>
using namespace std;

int main() {
    const double epsilon = 1e-12;

    Color c1, c2(255, 174, 10, Color::RGB), c3(0.2, 0.8, 0.4, Color::NORMALIZED_RGB);

    assert(c1.getRed() == 0);
    assert(c1.getGreen() == 0);
    assert(c1.getBlue() == 0);

    assert(c2.getRed() == 255);
    assert(c2.getGreen() == 174);
    assert(c2.getBlue() == 10);

    assert(abs(c3.getNormRed() - 0.2) < epsilon);
    assert(abs(c3.getNormGreen() - 0.8) < epsilon);
    assert(abs(c3.getNormBlue() - 0.4) < epsilon);

    c1.setRed(255);
    c1.setGreen(174);
    c1.setBlue(10);

    assert(c1.getRed() == 255);
    assert(c1.getGreen() == 174);
    assert(c1.getBlue() == 10);
}
