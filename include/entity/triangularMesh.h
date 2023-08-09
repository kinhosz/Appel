#ifndef ENTITY_TRIANGLEMESH_H
#define ENTITY_TRIANGLEMESH_H

#include <vector>  
#include <graphic/color.h>
#include <geometry/surfaceIntersection.h>
#include <geometry/ray.h>
#include <entity/box.h>  

class TriangularMesh : public Box {
private:
    int numTriangles;
    int numVertices;
    std::vector<Point> vertices;
    std::vector<Vetor> normals;
    std::vector<std::array<int, 3>> triangleIndices;
    std::vector<Vetor> triangleNormals;
    std::vector<Vetor> vertexNormals;
    std::vector<Color> colors; 

public:
    TriangularMesh();
    TriangularMesh(double kd, double ks, double ka, double kr, double kt, double roughness);
};

#endif
