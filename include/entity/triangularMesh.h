#ifndef ENTITY_TRIANGULARMESH_H
#define ENTITY_TRIANGULARMESH_H

#include <array>
#include <vector>  
#include <entity/box.h>
#include <graphic/color.h>
#include <geometry/point.h>
#include <geometry/vetor.h>

class TriangularMesh : public Box {
private:
    std::vector<Point>::size_type numberOfTriangles;
    std::vector<Point>::size_type numberOfVertices;
    std::vector<Point> vertices;
    std::vector<std::array<int, 3>> triangles;
    std::vector<Vetor> triangleNormals;
    std::vector<Vetor> vertexNormals;
    std::vector<Color> colors; 

public:
    TriangularMesh();
    TriangularMesh(
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
    );
};

#endif
