#include <geometry/point.h>
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
    return x > other.x && y > other.y && z > other.z;
}

bool Point::operator<(const Point &other) const {
    return x < other.x && y < other.y && z < other.z;
}

bool Point::operator==(const Point &other) const {
    return x == other.x && y == other.y && z == other.z;
}

bool Point::operator!=(const Point &other) const {
    return !(*this == other);
}