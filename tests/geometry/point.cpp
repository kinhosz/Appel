#include <geometry/point.h>
#include <assert.h>
#include <cmath>
using namespace std;

int main(){
    Point p1(1, 2, 3), p2(4, 5, 6);

    assert(abs(p1.distance(p2) - 5.196152) < 0.000001);
    assert(p1 == Point(1, 2, 3));
    assert(p2 == Point(4, 5, 6));
    assert(p1 != p2);
    assert(p1 < p2);
    assert(p2 > p1);
}
