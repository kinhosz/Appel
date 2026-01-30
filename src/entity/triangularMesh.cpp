#include <Appel/entity/triangularMesh.h>
#include <Appel/geometry/triangle.h>
#include <Appel/datastructure/wavefront.h>
#include <fstream>

namespace Appel {
    TriangularMesh::TriangularMesh() : Box() {
        this->triangles =  std::vector<Triangle>();
        this->_hasTexture = false;
    }

    TriangularMesh::TriangularMesh(std::vector<Triangle> triangles) : Box() {
        this->triangles = triangles;
        this->_hasTexture = false;
    }

    TriangularMesh::TriangularMesh(std::string filename) : Box() {
        this->_hasTexture = false;
        Wavefront wf(filename);
        triangles = wf.getTriangles();
    }

    void TriangularMesh::setTexture(std::string filename) {
        _hasTexture = texture.loadImage(filename);
    }

    bool TriangularMesh::hasTexture() const {
        return _hasTexture;
    }

    Pixel TriangularMesh::getTexture(double x, double y) const {
        int px = (texture.getWidth() - 1) * x;
        int py = (texture.getHeight() - 1) * y;
        return texture.getPixel(px, py);
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

    Point TriangularMesh::getPosition() const {
        return position;
    }

    void TriangularMesh::moveTo(const Point &p) {
        this->position = p;
    }

    Quaternion TriangularMesh::getRotation() const {
        return rotation;
    }

    void TriangularMesh::setRotation(const Quaternion &q) {
        rotation = q;
    }
}
