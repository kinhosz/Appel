#include <Point.h>
#include <cmath>

Point::Point() : x(0), y(0), z(0) {}

Point::Point(double x, double y, double z) : x(x), y(y), z(z) {}

double Point::distance(const Point &other) {
    double x_squared = (x - other.x) * (x - other.x);
    double y_squared = (y - other.y) * (y - other.y);
    double z_squared = (z - other.z) * (z - other.z);

    return sqrt(x_squared + y_squared + z_squared);
}