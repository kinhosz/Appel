#include <geometry/vetor.h>
#include <geometry/utils.h>
#include <math.h>

Vetor::Vetor(): x(0), y(0), z(0) {}

Vetor::Vetor(Point p): x(p.x), y(p.y), z(p.z) {}

Vetor::Vetor(double x, double y, double z): x(x), y(y), z(z) {}

Point Vetor::getPoint() const {
    return Point(x, y, z);
}

double Vetor::dot(const Vetor &other) const {
    return x*other.x + y*other.y + z*other.z;
}

Vetor Vetor::cross(const Vetor &other) const {
    double i = (y * other.z) - (z * other.y);
    double j = (z * other.x) - (x * other.z);
    double k = (x * other.y) - (y * other.x);

    return Vetor(i, j, k);
}

double Vetor::angle(const Vetor &other) const {
    double cost = dot(other) / (norm() * other.norm());
    cost = std::min(std::max(cost, -1.0 + EPSILON), 1.0 - EPSILON);

    return acos(cost);
}

double Vetor::norm() const {
    return sqrt(x*x + y*y + z*z);
}

Vetor Vetor::normalize() const {
    double n = norm();
    return Vetor(x/n, y/n, z/n);
}

Vetor Vetor::rotateX(double alpha) const {
    double ry = cos(alpha) * y - sin(alpha) * z;
    double rz = sin(alpha) * y + cos(alpha) * z;

    return Vetor(x, ry, rz);
}

Vetor Vetor::rotateY(double alpha) const {
    double rx = cos(alpha) * x + sin(alpha) * z;
    double rz = -sin(alpha) * x + cos(alpha) * z;

    return Vetor(rx, y, rz);
}

Vetor Vetor::rotateZ(double alpha) const {
    double rx = cos(alpha) * x - sin(alpha) * y;
    double ry = sin(alpha) * x + cos(alpha) * y;

    return Vetor(rx, ry, z);
}

Vetor Vetor::operator+(const Point &other) const {
    return Vetor(x + other.x, y + other.y, z + other.z);
}

Vetor Vetor::operator+(const Vetor &other) const {
    return Vetor(x + other.x, y + other.y, z + other.z);
}

Vetor Vetor::operator-(const Vetor &other) const {
    return Vetor(x - other.x, y - other.y, z - other.z);
}

Vetor Vetor::operator*(double p) const {
    return Vetor(x * p, y * p, z * p);
}

Vetor Vetor::operator/(double p) const {
    return Vetor(x / p, y / p, z / p);
}

bool Vetor::operator==(const Vetor &other) const {  
    return cmp(x, other.x) == 0 && cmp(y, other.y) == 0 && cmp(z, other.z) == 0;
}

bool Vetor::operator!=(const Vetor &other) const {
    return !(*this == other);
}

bool Vetor::isOrthogonal(const Vetor &other) const {
    return cmp(angle(other), PI/2.0) == 0;
}

bool Vetor::isParallel(const Vetor &other) const {
    return cmp(angle(other), 0.0) == 0;
}
