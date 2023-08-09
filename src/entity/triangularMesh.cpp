#include <entity/triangularMesh.h>

TriangularMesh::TriangularMesh() {
    numTriangles = 0;
    numVertices = 0;
}

TriangularMesh::TriangularMesh(double kd, double ks, double ka, double kr, double kt, double roughness)
    : Box(kd, ks, ka, kr, kt, roughness) {
    numTriangles = 0;
    numVertices = 0;
}

// void TriangularMesh::setVertices(const std::vector<Point>& vertexList) {
//     vertices = vertexList;
//     numVertices = static_cast<int>(vertexList.size());
// }


