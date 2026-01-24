#include <Appel/entity/triangularMesh.h>
#include <Appel/geometry/ray.h>
#include <Appel/geometry/utils.h>
#include <cassert>
#include <Appel/geometry/utils.h>
using namespace std;
using namespace Appel;

int main() {
    std::vector<Point> vertices = {
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(0, 1, 0),
        Point(1, 1, 0)
    };

    std::vector<std::array<int, 3>> triangles = {
        {0, 1, 2},
        {1, 2, 3}
    };

    std::vector<Vetor> vertexNormals = {
        Vetor(0, 0, 1),  
        Vetor(0, 0, 1),  
        Vetor(0, 0, 1),  
        Vetor(0, 0, 1)   
    };

    std::vector<Vetor> triangleNormals = {
        Vetor(0, 0, 1),  
        Vetor(0, 0, 1)
    };

    std::vector<Color> colors = {
        Color(0.1, 0.2, 0.3),
        Color(0.4, 0.5, 0.6)
    };

    std::vector<Triangle> createdTriangles = createTriangles(
        triangles.size(),
        vertices,
        triangles,
        vertexNormals,
        triangleNormals,
        colors
    );

    TriangularMesh mesh(createdTriangles);

    mesh.setPhongValues(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);

    Vetor vector(1, 0, 0);
    Point point(0, 0, 0);
    Ray ray(point, vector);

    SurfaceIntersection sf = mesh.intersect(ray);

    assert(sf.color == Color(0, 0, 0));
    assert(cmp(sf.distance, DOUBLE_INF) == 0);
    assert(sf.normal == Vetor(0, 0, 1));

    return 0;
}
