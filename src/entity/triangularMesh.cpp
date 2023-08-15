#include <entity/triangularMesh.h>
#include <geometry/triangle.h>
#include <graphic/color.h>
#include <assert.h>

TriangularMesh::TriangularMesh() : Box() {
    this->triangles =  std::vector<Triangle>();
}

Triangle TriangularMesh::getTriangle(int index) {
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
