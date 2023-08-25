#include <entity/triangularMesh.h>
#include <geometry/triangle.h>

TriangularMesh::TriangularMesh() : Box() {
    this->triangles =  std::vector<Triangle>();
}

Triangle TriangularMesh::getTriangle(int index) const {
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

SurfaceIntersection TriangularMesh::intersect(Ray ray) const {
    SurfaceIntersection nearSurface;

    for(Triangle triangle: triangles) {
        SurfaceIntersection surface = triangle.intersect(ray);

        if(surface.distance < nearSurface.distance) nearSurface = surface;
    }

    return nearSurface;
}
