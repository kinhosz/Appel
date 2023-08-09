#include <entity/triangularMesh.h>
#include <assert.h>

TriangularMesh::TriangularMesh() : Box() {
    this->numberOfTriangles = 0;
    this->numberOfVertices = 0;
    this->vertices = std::vector<Point>();
    this->triangles = std::vector<std::array<int, 3>>();
    this->triangleNormals = std::vector<Vetor>();
    this->vertexNormals = std::vector<Vetor>();
    this->colors = std::vector<Color>();
}

TriangularMesh::TriangularMesh(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point>::size_type numberOfVertices,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Vetor> triangleNormals,
    std::vector<Vetor> vertexNormals,
    std::vector<Color> colors,
    double kd,
    double ks,
    double ka,
    double kr,
    double kt,
    double roughness
) : Box(kd, ks, ka, kr, kt, roughness) {
    assert(numberOfTriangles == triangles.size());
    assert(numberOfVertices == vertices.size());
    this->numberOfTriangles = numberOfTriangles;
    this->numberOfVertices = numberOfVertices;
    this->vertices = vertices;
    this->triangles = triangles;
    this->triangleNormals = triangleNormals;
    this->vertexNormals = vertexNormals;
    this->colors = colors;
}
