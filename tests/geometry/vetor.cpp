#include <Appel/geometry/vetor.h>
#include <Appel/geometry/utils.h>
#include <assert.h>
#include <cmath>
using namespace std;

int main(){
    const double epsilon = 1e-6;

    Vetor v1(1, 2, 3), v2(4, 5, 6);

    assert(abs(v1.dot(v2) - 32) < epsilon);
    assert(v1.cross(v2) == Vetor(-3, 6, -3));
    assert(abs(v1.angle(v2) - 0.225726) < epsilon);
    assert(abs(v1.norm() - 3.741657) < epsilon);

    assert(v1.normalize() == Vetor(0.267261241912424397, 0.534522483824848793, 0.80178372573727319));
    assert(v2.normalize() == Vetor(0.455842305838551787, 0.569802882298189761, 0.683763458757827625));

    assert(v1.rotateX(PI/2.0) == Vetor(1, -3, 2));
    assert(v1.rotateY(PI/2.0) == Vetor(3, 2, -1));
    assert(v1.rotateZ(PI/2.0) == Vetor(-2, 1, 3));
    
    assert(v2.rotateX(PI/2) == Vetor(4, -6, 5));
    assert(v2.rotateY(PI/2) == Vetor(6, 5, -4));
    assert(v2.rotateZ(PI/2) == Vetor(-5, 4, 6));

    assert(v1 + v2 == Vetor(5, 7, 9));
    assert(v1 - v2 == Vetor(-3, -3, -3));
    assert(v1 * 2 == Vetor(2, 4, 6));
    assert(v1 / 2 == Vetor(0.5, 1, 1.5));
    assert(v1 == Vetor(1, 2, 3));
    assert(v2 == Vetor(4, 5, 6));
    assert(v1 != v2);
}
