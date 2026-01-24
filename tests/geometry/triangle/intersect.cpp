#include <Appel/geometry/triangle.h>
#include <Appel/geometry/utils.h>
using namespace std;
using namespace Appel;

int main(){
    // using eq = 2x - 3y + z - 4 = 0
    Point p1(1, 0, 2);
    Point p2(0, 1, 7);
    Point p3(-1, -1, 3);

    Vetor normal = (Vetor(p2) - Vetor(p1)).cross(Vetor(p3) - Vetor(p1)).normalize();

    Triangle triangle(p1, p2, p3, Vetor(0, 0, 1), Vetor(0, 0, 1), Vetor(0, 0, 1), normal, Color(1, 1, 1));

    Point insideTriangle(0, 0, 4);
    Point outsideTriangle(-1, 0, 6);

    Ray r1(Point(0, 0, 0), Vetor(0, 0, 1));
    Ray r2(Point(0, 0, 0), Vetor(0, 0, -1));
    Ray r3(Point(-1, 0, 0), Vetor(0, 0, 1));

    SurfaceIntersection sf1, sf2, sf3;

    sf1 = triangle.intersect(r1);
    sf2 = triangle.intersect(r2);
    sf3 = triangle.intersect(r3);

    assert(cmp(sf1.distance, DOUBLE_INF) == -1);
    assert(sf1.color == triangle.color);
    assert(sf1.normal == triangle.getTriangleNormal());

    assert(cmp(sf2.distance, DOUBLE_INF) == 0);
    assert(sf2.color == Color(0, 0, 0));

    assert(cmp(sf3.distance, DOUBLE_INF) == 0);
    assert(sf3.color == Color(0, 0, 0));
}
