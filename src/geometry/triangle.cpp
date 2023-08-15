#include <geometry/triangle.h>
#include <geometry/utils.h>
#include <graphic/color.h>
#include <math.h>

Triangle::Triangle() {
    vertices[0] = Point();
    vertices[1] = Point();
    vertices[2] = Point();
    color = Color();
    triangleNormal = Vector();
    normals[0] = Vector();
    normals[1] = Vector();
    normals[2] = Vector();
}

Triangle::Triangle(Point v1, Point v2, Point v3, Vector n1, Vector n2, Vector n3, Vector triangleNormal, Color triangleColor) {
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    color = triangleColor;
    this->triangleNormal = triangleNormal;
    normals[0] = n1;
    normals[1] = n2;
    normals[2] = n3;
}

double Triangle::area() const {
    Vetor side1 = Vetor(vertices[1]) - Vetor(vertices[0]);
    Vetor side2 = Vetor(vertices[2]) - Vetor(vertices[0]);
    Vetor cross_product = side1.cross(side2);
    return 0.5 * cross_product.norm();
}

Point Triangle::centroid() const {
    Point centroid;
    for (int i = 0; i < 3; ++i) {
        centroid.x += vertices[i].x;
        centroid.y += vertices[i].y;
        centroid.z += vertices[i].z;
    }
    centroid.x /= 3.0;
    centroid.y /= 3.0;
    centroid.z /= 3.0;
    return centroid;
}

bool Triangle::operator==(const Triangle& other) const {
    return vertices[0] == other.vertices[0] &&
           vertices[1] == other.vertices[1] &&
           vertices[2] == other.vertices[2];
}

bool Triangle::operator!=(const Triangle& other) const {
    return !(*this == other);
}