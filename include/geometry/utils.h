#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H
#include <float.h>
#include <vector>
#include <array>
#include <geometry/triangle.h> 
#include <geometry/point.h>
#include <graphic/color.h>

#define PI 3.1415926535897932384
#define DOUBLE_INF DBL_MAX

extern const double EPSILON;
extern int cmp(double a, double b);

std::vector<Triangle> createTriangles(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Vetor> vertexNormals,  
    std::vector<Vetor> triangleNormals,
    std::vector<Color> colors
);

#endif