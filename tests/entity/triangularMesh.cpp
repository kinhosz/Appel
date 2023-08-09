#include <entity/triangularMesh.h>
#include <geometry/ray.h>
#include <geometry/utils.h>
#include <cassert>
using namespace std;

int main() {
    TriangularMesh mesh;
    mesh.setDiffuseCoefficient(0.1);
    mesh.setSpecularCoefficient(0.2);
    mesh.setAmbientCoefficient(0.3);
    mesh.setReflectionCoefficient(0.4);
    mesh.setTransmissionCoefficient(0.5);
    mesh.setRoughnessCoefficient(0.6);

    Vetor vector(1, 0, 0);
    Point point(0, 0, 0);
    Ray ray(point, vector);

    SurfaceIntersection sf = mesh.intersect(ray);

    assert(sf.color == Color(0, 0, 0));
    assert(cmp(sf.distance, DOUBLE_INF) == 0);
    assert(sf.normal == Vetor(0, 0, 1));

    TriangularMesh defaultMesh;

    return 0;
}
