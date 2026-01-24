#include <Appel/entity/plane.h>
#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>
#include <Appel/graphic/color.h>
#include <assert.h>
using namespace std;
using namespace Appel;

int main() {
    Plane p1, p2(Point(0.00, 0.00, 0.00), Vetor(0.00, 1.00, 0.00), Color(0.23, 0.60, 0.98));

    p2.setPhongValues(0.60, 0.80, 0.20, 0.10, 0.00, 0.50);

    assert(p1.getPoint() == Point(0.00, 0.00, 0.00));
    assert(p1.getNormalVector() == Vetor(0.00, 0.00, 0.00));
    assert(p1.getColor().getRed() == 0.00);
    assert(p1.getColor().getGreen() == 0.00);
    assert(p1.getColor().getBlue() == 0.00);

    assert(p2.getPoint() == Point(0.00, 0.00, 0.00));
    assert(p2.getNormalVector() == Vetor(0.00, 1.00, 0.00));
    assert(p2.getColor().getRed() == 0.23);
    assert(p2.getColor().getGreen() == 0.60);
    assert(p2.getColor().getBlue() == 0.98);

    p1.setPoint(Point(1.00, 1.00, 1.00));
    p1.setNormalVector(Vetor(1.00, 0.00, 0.00));
    p1.setColor(Color(0.63, 0.96, 0.68));

    assert(p1.getPoint() == Point(1.00, 1.00, 1.00));
    assert(p1.getNormalVector() == Vetor(1.00, 0.00, 0.00));
    assert(p1.getColor().getRed() == 0.63);
    assert(p1.getColor().getGreen() == 0.96);
    assert(p1.getColor().getBlue() == 0.68);
}
