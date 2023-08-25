#include <geometry/triangle.h>
#include <geometry/utils.h>
#include <graphic/color.h>
#include <geometry/vetor.h> 
#include <geometry/surfaceIntersection.h>
#include <geometry/ray.h>
#include <math.h>

Triangle::Triangle() {
    vertices[0] = Point();
    vertices[1] = Point();
    vertices[2] = Point();
    color = Color();
    triangleNormal = Vetor();
    vertexNormals[0] = Vetor();
    vertexNormals[1] = Vetor();
    vertexNormals[2] = Vetor();
}

Triangle::Triangle(Point v1, Point v2, Point v3, Vetor n1, Vetor n2, Vetor n3, Vetor triangleNormal, Color triangleColor) {
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    color = triangleColor;
    this->triangleNormal = triangleNormal;
    vertexNormals[0] = n1;
    vertexNormals[1] = n2;
    vertexNormals[2] = n3;
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

bool Triangle::isInside(const Point &p) const {
    std::vector<std::vector<double>> matrix = {
        {vertices[0].x, vertices[1].x, vertices[2].x, p.x},
        {vertices[0].y, vertices[1].y, vertices[2].y, p.y},
        {vertices[0].z, vertices[1].z, vertices[2].z, p.z},
        {1.0, 1.0, 1.0, 1.0}
    };

    std::vector<double> baricenter = gaussElimination(matrix);

    if(baricenter.size() != 3) return false;
    assert(cmp(baricenter[0] + baricenter[1] + baricenter[2], 1.0) == 0);

    bool isInside = true;

    for(int i=0;i<3;i++){
        if(cmp(baricenter[i], 0) == -1 || cmp(baricenter[i], 1) == 1) isInside = false;
    }

    return isInside;
}

SurfaceIntersection Triangle::intersect(Ray ray) const {
    Vetor normal = triangleNormal.normalize();

    if(normal.isOrthogonal(ray.direction)) return SurfaceIntersection();

    double D = -normal.x * vertices[0].x - normal.y * vertices[0].y - normal.z * vertices[0].z;
    double A = normal.x, B = normal.y, C = normal.z;

    double c1 = (A * ray.location.x + B * ray.location.y + C * ray.location.z + D);
    double c2 = (A * ray.direction.x + B * ray.direction.y + C * ray.direction.z);

    double t = -c1/c2;

    if(cmp(t, 0) == -1) return SurfaceIntersection();

    Point matchedPoint = ray.pointAt(t);

    if(!isInside(matchedPoint)) return SurfaceIntersection();

    return SurfaceIntersection(color, t, normal);
}
