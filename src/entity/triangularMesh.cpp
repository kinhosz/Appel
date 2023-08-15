#include <entity/triangularMesh.h>
#include <assert.h>

std::vector<Triangle> createTriangles(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Color> colors
) {
    std::vector<Triangle> triangleObjects;
    for (std::vector<Point>::size_type i = 0; i < numberOfTriangles; ++i) {
        const std::array<int, 3>& indices = triangles[i];
        Triangle triangle(
            vertices[indices[0]],
            vertices[indices[1]], 
            vertices[indices[2]],
            colors[i]
        );
        triangleObjects.push_back(triangle);
    }
    return triangleObjects;
}

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
            this->vertices.insert(&triangle.getVertex(i));
        }
    }

    for (const Triangle& triangle : triangles) {
        for (int i = 0; i < 3; ++i) {
            this->triangleNormals.insert(&triangle.normal());
        }
    }

}
