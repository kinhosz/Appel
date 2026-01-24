#include <Appel/geometry/ray.h>
#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>
#include <Appel/geometry/utils.h>
#include <assert.h>
using namespace std;

int main() {
    Point p(0, 0, 0);
    Vetor v(1, 0, 0);

    Ray ray(p, v);

    Point at = ray.pointAt(2.0);

    assert(cmp(at.x, 2.0) == 0);
    assert(cmp(at.y, 0.0) == 0);
    assert(cmp(at.z, 0.0) == 0);
}
