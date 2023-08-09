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
    int numberOfTriangles;
    int numberOfVertices;
    std::vector<Point> vertices;
    std::vector<std::array<int, 3>> triangleIndices;
    std::vector<Vetor> triangleNormals;
    std::vector<Vetor> vertexNormals;
    std::vector<Color> colors; 

public:
    TriangularMesh();
    TriangularMesh(
        int numberOfTriangles,
        int numberOfVertices,
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
