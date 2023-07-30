#include <geometry/point.h>
#include <geometry/utils.h>
#include <cmath>

Point::Point() : x(0), y(0), z(0) {}

Point::Point(double x, double y, double z) : x(x), y(y), z(z) {}

double Point::distance(const Point &other) const {
    double xSquared = (x - other.x) * (x - other.x);
    double ySquared = (y - other.y) * (y - other.y);
    double zSquared = (z - other.z) * (z - other.z);

    return sqrt(xSquared + ySquared + zSquared);
}

bool Point::operator>(const Point &other) const {
    if(cmp(x, other.x) != 0) return cmp(x, other.x) == 1;
    if(cmp(y, other.y) != 0) return cmp(y, other.y) == 1;
    return cmp(z, other.z) == 1;
}

bool Point::operator<(const Point &other) const {
    if(cmp(x, other.x) != 0) return cmp(x, other.x) == -1;
    if(cmp(y, other.y) != 0) return cmp(y, other.y) == -1;
    return cmp(z, other.z) == -1;
}

bool Point::operator==(const Point &other) const {
    return cmp(x, other.x) == 0 && cmp(y, other.y) == 0 && cmp(z, other.z) == 0;
}

bool Point::operator!=(const Point &other) const {
    return !(*this == other);
}