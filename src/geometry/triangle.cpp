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

Triangle::Triangle(Point v1, Point v2, Point v3, Color triangleColor) {
    vertices[0] = v1;
    vertices[1] = v2;
    vertices[2] = v3;
    color = triangleColor;
    
    this->triangleNormal = (Vetor(v2) - Vetor(v1)).cross(Vetor(v3) - Vetor(v1)).normalize();
    vertexNormals[0] = vertexNormals[1] = vertexNormals[2] = Vetor(0, 0, 1);
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

Point Triangle::getVertex(int index) const {
    assert(index >= 0 && index < 3);
    return vertices[index];
}

Vetor Triangle::getVertexNormal(int index) const {
    assert(index >= 0 && index < 3);
    return vertexNormals[index];
}

Vetor Triangle::getTriangleNormal() const {
    return triangleNormal;
}

Point Triangle::getMin() const {
    return Point(
        std::min(vertices[0].x, std::min(vertices[1].x, vertices[2].x)),
        std::min(vertices[0].y, std::min(vertices[1].y, vertices[2].y)),
        std::min(vertices[0].z, std::min(vertices[1].z, vertices[2].z))
    );
}

Point Triangle::getMax() const {
    return Point(
        std::max(vertices[0].x, std::max(vertices[1].x, vertices[2].x)),
        std::max(vertices[0].y, std::max(vertices[1].y, vertices[2].y)),
        std::max(vertices[0].z, std::max(vertices[1].z, vertices[2].z))
    );
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
    Vetor A = vertices[0];
    Vetor B = vertices[1];
    Vetor C = vertices[2];
    Vetor P = p;

    double ABC = 0.5 * ((B - A).cross(C - A)).norm();
    double PBC = 0.5 * (((B - P).cross(C - P))).norm();
    double PAC = 0.5 * (((A - P).cross(C - P))).norm();
    double PAB = 0.5 * (((A - P).cross(B - P))).norm();

    if(cmp(ABC, 0.0) == 0) return false;

    double u = PBC / ABC;
    double v = PAC / ABC;
    double w = PAB / ABC;

    double sumt = u + v + w;

    if(cmp(sumt, 1.0) != 0) return false;
    if(cmp(u, 0.0) == -1 || cmp(u, 1.0) == 1) return false;
    if(cmp(v, 0.0) == -1 || cmp(v, 1.0) == 1) return false;
    if(cmp(w, 0.0) == -1 || cmp(w, 1.0) == 1) return false;

    return true;
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

SurfaceIntersection Triangle::getSurface(Ray ray) const {
    Vetor normal = triangleNormal.normalize();

    if(normal.isOrthogonal(ray.direction)) return SurfaceIntersection();

    double D = -normal.x * vertices[0].x - normal.y * vertices[0].y - normal.z * vertices[0].z;
    double A = normal.x, B = normal.y, C = normal.z;

    double c1 = (A * ray.location.x + B * ray.location.y + C * ray.location.z + D);
    double c2 = (A * ray.direction.x + B * ray.direction.y + C * ray.direction.z);

    double t = -c1/c2;

    if(cmp(t, 0) == -1) return SurfaceIntersection();

    return SurfaceIntersection(color, t, normal);
}
