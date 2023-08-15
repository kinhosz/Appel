#ifndef ENTITY_TRIANGULARMESH_H
#define ENTITY_TRIANGULARMESH_H

#include <vector>  
#include <entity/box.h>
#include <geometry/point.h>
#include <geometry/vetor.h>
#include <geometry/triangle.h>

class TriangularMesh : public Box {
private:
    std::vector<Triangle> triangles;

public:
    TriangularMesh();

    TriangularMesh(
        const std::vector<Triangle> triangles, 
        double kd,
        double ks,
        double ka,
        double kr,
        double kt,
        double roughness
    );

    Triangle getTriangle(int index) const;
};

#endif
