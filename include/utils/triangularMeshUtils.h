#ifndef TRIANGULAR_MESH_UTILS_H
#define TRIANGULAR_MESH_UTILS_H

#include <vector>
#include <array>
#include <geometry/triangle.h> 
#include <geometry/point.h>
#include <graphic/color.h>

std::vector<Triangle> createTriangles(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Color> colors
);

#endif 