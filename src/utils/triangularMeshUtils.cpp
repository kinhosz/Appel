#include <triangularMeshUtils.h>
#include <geometry/triangle.h>
#include <geometry/point.h>
#include <graphic/color.h>

std::vector<Triangle> createTriangles(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Color> colors
) {
    std::vector<Triangle> triangleObjects;
    for (std::vector<Point>::size_type i = 0; i < numberOfTriangles; ++i) {
        const std::array<int, 3>& indices = triangles[i];
        Triangle triangle(
            vertices[indices[0]],
            vertices[indices[1]], 
            vertices[indices[2]],
            colors[i]
        );
        triangleObjects.push_back(triangle);
    }
    return triangleObjects;
}
