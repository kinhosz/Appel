#include <graphic/color.h>
#include <assert.h>
#include <cmath>
using namespace std;

int main() {
    Color c1, c2(0.25, 0.83, 0.42);

    assert(c1.getRed() == 0.00);
    assert(c1.getGreen() == 0.00);
    assert(c1.getBlue() == 0.00);

    assert(c2.getRed() == 0.25);
    assert(c2.getGreen() == 0.83);
    assert(c2.getBlue() == 0.42);

    c1.setRed(1.00);
    c1.setGreen(0.68);
    c1.setBlue(0.04);
    
    assert(c1.getRed() == 1.00);
    assert(c1.getGreen() == 0.68);
    assert(c1.getBlue() == 0.04);
}
