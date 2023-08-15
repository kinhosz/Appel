#include <entity/triangularMesh.h>
#include <graphic/color.h>
#include <assert.h>

TriangularMesh::TriangularMesh() : Box() {
    this->vertices = std::vector<Point>();
    this->triangles =  std::vector<Triangle>();
    this->triangleNormals = std::vector<Vetor>();
}

TrianugularMesh::getTriangle(int index) {
    return this->triangles[index];
}

TriangularMesh::TriangularMesh(
    std::vector<Triangle> triangles,
    double kd,
    double ks,
    double ka,
    double kr,
    double kt,
    double roughness
) : Box(kd, ks, ka, kr, kt, roughness) {
    this->triangles = triangles;
}
