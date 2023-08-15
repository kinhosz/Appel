#include <geometry/triangle.h>
#include <geometry/utils.h>
#include <graphic/color.h>
#include <math.h>

Triangle::Triangle() {
    vertices[0] = Point();
    vertices[1] = Point();
    vertices[2] = Point();
}

Triangle::Triangle(Point v1, Point v2, Point v3, Color triangleColor) {
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    
    color = triangleColor;
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

Vetor Triangle::normal() const {
    Vetor side1 = Vetor(vertices[1]) - Vetor(vertices[0]);
    Vetor side2 = Vetor(vertices[2]) - Vetor(vertices[0]);
    return side1.cross(side2).normalize();
}

bool Triangle::contains(const Point& p) const {
    Vetor edge0 = Vetor(vertices[1]) - Vetor(vertices[0]);
    Vetor edge1 = Vetor(vertices[2]) - Vetor(vertices[1]);
    Vetor edge2 = Vetor(vertices[0]) - Vetor(vertices[2]);

    double dot00 = edge0.dot(edge0);
    double dot01 = edge0.dot(edge1);
    double dot02 = edge0.dot(edge2);
    double dot11 = edge1.dot(edge1);
    double dot12 = edge1.dot(edge2);

    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v < 1);
}

bool Triangle::operator==(const Triangle& other) const {
    return vertices[0] == other.vertices[0] &&
           vertices[1] == other.vertices[1] &&
           vertices[2] == other.vertices[2];
}

bool Triangle::operator!=(const Triangle& other) const {
    return !(*this == other);
}