#ifndef ENTITY_TRIANGULARMESH_H
#define ENTITY_TRIANGULARMESH_H

#include <vector>
#include <string>
#include <entity/box.h>
#include <geometry/triangle.h>
#include <graphic/image.h>

class TriangularMesh : public Box {
private:
    std::vector<Triangle> triangles;
    bool _hasTexture;
    Image texture;
public:
    TriangularMesh();
    TriangularMesh(const std::vector<Triangle> triangles);
    TriangularMesh(std::string filename);

    bool hasTexture() const;
    void setTexture(std::string filename);
    Pixel getTexture(double x, double y) const;
    Triangle getTriangle(int index) const;
    std::vector<Triangle> getTriangles() const;
    SurfaceIntersection intersect(const Ray& ray) const override;
};

#endif
