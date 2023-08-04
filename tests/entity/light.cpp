#include <entity/light.h>
#include <geometry/point.h>
#include <graphic/pixel.h>
#include <assert.h>
using namespace std;

int main(){
    Light l(Point(-1, 2, 10), Pixel(255, 15, 0));

    assert(l.getLocation() == Point(-1, 2, 10));
    assert(l.getIntensity().getRed() == 255);
    assert(l.getIntensity().getGreen() == 15);
    assert(l.getIntensity().getBlue() == 0);

    l.setLocation(Point(0, 0, 0));
    l.setIntensity(Pixel(0, 15, 255));

    assert(l.getLocation() == Point(0, 0, 0));
    assert(l.getIntensity().getRed() == 0);
    assert(l.getIntensity().getGreen() == 15);
    assert(l.getIntensity().getBlue() == 255);
}
