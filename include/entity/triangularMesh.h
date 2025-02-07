#ifndef ENTITY_TRIANGULARMESH_H
#define ENTITY_TRIANGULARMESH_H

#include <vector>  
#include <entity/box.h>
#include <geometry/triangle.h>

class TriangularMesh : public Box {
private:
    std::vector<Triangle> triangles;

public:
    TriangularMesh();
    TriangularMesh(const std::vector<Triangle> triangles);

    Triangle getTriangle(int index) const;
    std::vector<Triangle> getTriangles() const;
    SurfaceIntersection intersect(const Ray& ray) const override;
};

#endif
