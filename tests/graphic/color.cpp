#include <graphic/color.h>
#include <geometry/constants.h>
#include <assert.h>
#include <cmath>
using namespace std;

int main() {
    Color c1, c2(0, 255, 0);

    assert(c1.getRed() == 0);
    assert(c1.getGreen() == 0);
    assert(c1.getBlue() == 0);

    assert(c2.getRed() == 0);
    assert(c2.getGreen() == 255);
    assert(c2.getBlue() == 0);

    c1.setRed(255);
    c1.setGreen(255);
    c1.setBlue(255);

    c2.setRed(255);
    c2.setGreen(0);
    c2.setBlue(0);

    assert(c1.getRed() == 255);
    assert(c1.getGreen() == 255);
    assert(c1.getBlue() == 255);
    assert(abs(c1.getNormRed() - 1.0) < EPSILON);
    assert(abs(c1.getNormGreen() - 1.0) < EPSILON);
    assert(abs(c1.getNormBlue() - 1.0) < EPSILON);

    assert(c2.getRed() == 255);
    assert(c2.getGreen() == 0);
    assert(c2.getBlue() == 0);
    assert(abs(c2.getNormRed() - 1.0) < EPSILON);
    assert(abs(c2.getNormGreen() - 0.0) < EPSILON);
    assert(abs(c2.getNormBlue() - 0.0) < EPSILON);
}
