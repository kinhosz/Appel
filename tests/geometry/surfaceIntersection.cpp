#include <cassert>
#include <Appel/geometry/surfaceIntersection.h>
#include <Appel/geometry/utils.h>

using namespace std;
using namespace Appel;

int main(){
    SurfaceIntersection sf;
    
    assert(sf.color == Color(0, 0, 0));
    assert(cmp(sf.distance, DOUBLE_INF) == 0);
    assert(sf.normal == Vetor(0, 0, 1));

    Color color(1, 1, 1);
    double distance = 10;
    Vetor normal(2, 0, 0);

    SurfaceIntersection sf2(color, distance, normal);
    assert(sf2.color == Color(1, 1, 1));
    assert(cmp(sf2.distance, 10) == 0);
    assert(sf2.normal == Vetor(1, 0, 0));
}
