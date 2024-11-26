#include <entity/triangularMesh.h>
#include <geometry/triangle.h>

TriangularMesh::TriangularMesh() : Box() {
    this->triangles =  std::vector<Triangle>();
}


TriangularMesh::TriangularMesh(std::vector<Triangle> triangles) : Box() {
    this->triangles = triangles;
}

Triangle TriangularMesh::getTriangle(int index) const {
    return this->triangles[index];
}

std::vector<Triangle> TriangularMesh::getTriangles() const {
    return triangles;
}

SurfaceIntersection TriangularMesh::intersect(const Ray& ray) const {
    SurfaceIntersection nearSurface;

    for(Triangle triangle: triangles) {
        SurfaceIntersection surface = triangle.intersect(ray);

        if(surface.distance < nearSurface.distance) nearSurface = surface;
    }

    return nearSurface;
}
