#include <entity/box.h>
#include <geometry/ray.h>
#include <geometry/utils.h>
#include <cassert>
using namespace std;

int main(){
    Box b1(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), b2(0.6, 0.5, 0.4, 0.3, 0.2, 0.1);

    Vetor vetor(1, 0, 0);
    Point point(0, 0, 0);

    Ray ray(point, vetor);

    SurfaceIntersection sf = b1.intersect(ray);

    assert(sf.color == Color(0, 0, 0));
    assert(cmp(sf.distance, DOUBLE_INF) == 0);
    assert(sf.normal == Vetor(0, 0, 1));
}
