#include <entity/light.h>
#include <geometry/point.h>
#include <graphic/color.h>
#include <assert.h>
using namespace std;

int main(){
    Light l(Point(-1, 2, 10), Color(0.15, 0.97, 0.50));

    assert(l.getLocation() == Point(-1, 2, 10));
    assert(l.getIntensity().getRed() == 0.15);
    assert(l.getIntensity().getGreen() == 0.97);
    assert(l.getIntensity().getBlue() == 0.50);

    l.setLocation(Point(0, 0, 0));
    l.setIntensity(Color(1.00, 0.37, 0.24));

    assert(l.getLocation() == Point(0, 0, 0));
    assert(l.getIntensity().getRed() == 1.00);
    assert(l.getIntensity().getGreen() == 0.37);
    assert(l.getIntensity().getBlue() == 0.24);
}
