#include <Appel/geometry/point.h>
#include <assert.h>
#include <cmath>
using namespace std;

int main(){
    const double epsilon = 1e-6;

    Point p1(1, 2, 3), p2(4, 5, 6);

    assert(abs(p1.distance(p2) - 5.196152) < epsilon);
    assert(abs(p2.distance(Point(0, 0, 0)) - 8.774964) < epsilon);

    assert(p1 == Point(1, 2, 3));
    assert(p2 == Point(4, 5, 6));
    assert(p1 != p2);
    assert(p1 < p2);
    assert(p2 > p1);
}
