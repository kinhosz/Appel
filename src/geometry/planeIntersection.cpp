#include <geometry/planeIntersection.h>
#include <geometry/utils.h>

PlaneIntersection::PlaneIntersection() {
    color = Color(0, 0, 0);
    distance = DOUBLE_INF;
    normal = Vetor(0, 0, 1);
}

PlaneIntersection::PlaneIntersection(Color color, double distance, Vetor normal) {
    this->color = color;
    this->distance = distance;
    this->normal = normal.normalize();
}