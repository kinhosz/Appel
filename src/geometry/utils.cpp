#include <geometry/triangle.h>
#include <geometry/point.h>
#include <graphic/color.h>
#include <geometry/vetor.h> 
#include <geometry/utils.h>
#include <cmath>

const double EPSILON = 1e-10;

int cmp(double a, double b){
    if(std::abs(a - b) < EPSILON) return 0;
    return (a < b ? -1 : 1);
}

std::vector<Triangle> createTriangles(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Vetor> vertexNormals,  
    std::vector<Vetor> triangleNormals, 
    std::vector<Color> colors
) {
    std::vector<Triangle> triangleObjects;
    for (std::vector<Point>::size_type i = 0; i < numberOfTriangles; ++i) {
        const std::array<int, 3>& indices = triangles[i];
        Triangle triangle(
            vertices[indices[0]],
            vertices[indices[1]],
            vertices[indices[2]],
            vertexNormals[indices[0]],  
            vertexNormals[indices[1]],
            vertexNormals[indices[2]],
            triangleNormals[i], 
            colors[i]
        );
        triangleObjects.push_back(triangle);
    }
    return triangleObjects;
}

double getAngle(double x, double y) {
    if(cmp(x, 0.0) == 0 && cmp(y, 0.0) == 0) return 0;
    else if(cmp(x, 0.0) == 0) {
        if(cmp(y, 0.0) == 1) return PI/2.0;
        else return -PI/2.0;
    }
    else if(cmp(y, 0.0) == 0) {
        if(cmp(x, 0.0) == 1) return 0.0;
        else return PI;
    }
    return atan(y / x);
}
