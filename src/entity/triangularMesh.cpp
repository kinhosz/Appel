#include <entity/triangularMesh.h>
#include <graphic/color.h>
#include <assert.h>

TriangularMesh::TriangularMesh() : Box() {
    this->vertices = std::vector<Point>();
    this->triangles =  std::vector<Triangle>();
    this->triangleNormals = std::vector<Vetor>();
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

    for (const Triangle& triangle : triangles) {
        for (int i = 0; i < 3; ++i) {
            this->vertices.push_back(triangle.getVertex(i));
        }
    }

    for (const Triangle& triangle : triangles) {
        for (int i = 0; i < 3; ++i) {
            this->triangleNormals.push_back(triangle.normal());
        }
    }

}
