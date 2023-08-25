#include <geometry/triangle.h>
using namespace std;

int main(){
    // using eq = 2x - 3y + z - 4 = 0
    Point p1(1, 0, 2);
    Point p2(0, 1, 7);
    Point p3(-1, -1, 3);

    Vetor normal = (Vetor(p2) - Vetor(p1)).cross(Vetor(p3) - Vetor(p1)).normalize();

    Triangle triangle(p1, p2, p3, Vetor(0, 0, 1), Vetor(0, 0, 1), Vetor(0, 0, 1), normal, Color(1, 1, 1));

    Point insideTriangle(0, 0, 4);
    Point outsideTriangle(-1, 0, 6);

    assert(triangle.isInside(insideTriangle));
    assert(!triangle.isInside(outsideTriangle));
}
