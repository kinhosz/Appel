#include <Point.h>
#include <cmath>

Point::Point() : x(0), y(0) {}

Point::Point(double x, double y) : x(x), y(y) {}

double Point::distance(const Point &other) {
    return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
}