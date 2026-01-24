#include <Appel/entity/sphere.h>
#include <Appel/geometry/point.h>
#include <Appel/graphic/color.h>
#include <assert.h>
using namespace std;
using namespace Appel;

int main() {
    Sphere s1, s2(Point(1.55, -2.05, 3.00), 8.00, Color(0.23, 0.60, 0.98));

    s2.setPhongValues(0.40, 0.92, 1.00, 0.11, 0.24, 0.31);

    assert(s1.getCenter() == Point(0.0, 0.0, 0.0));
    assert(s1.getRadius() == 0.0);
    assert(s1.getColor().getRed() == 0.0);
    assert(s1.getColor().getGreen() == 0.0);
    assert(s1.getColor().getBlue() == 0.0);

    assert(s2.getCenter() == Point(1.55, -2.05, 3.00));
    assert(s2.getRadius() == 8.0);
    assert(s2.getColor().getRed() == 0.23);
    assert(s2.getColor().getGreen() == 0.60);
    assert(s2.getColor().getBlue() == 0.98);

    s1.setCenter(Point(3.55, 1.05, 2.00));
    s1.setRadius(4.50);
    s1.setColor(Color(0.63, 0.96, 0.68));

    assert(s1.getCenter() == Point(3.55, 1.05, 2.00));
    assert(s1.getRadius() == 4.50);
    assert(s1.getColor().getRed() == 0.63);
    assert(s1.getColor().getGreen() == 0.96);
    assert(s1.getColor().getBlue() == 0.68);
}
