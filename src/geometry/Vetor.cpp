#include <Vetor.h>
#include <math.h>

Vetor::Vetor(): x(0), y(0), z(0) {}

Vetor::Vetor(double x, double y, double z): x(x), y(y), z(z) {}

double Vetor::dot(const Vetor &other) const {
    return x*other.x + y*other.y + z*other.z;
}

Vetor& Vetor::cross(const Vetor &other) const {
    double i = (y * other.z) - (z * other.y);
    double j = (z * other.x) - (x * other.z);
    double k = (x * other.y) - (y * other.x);

    return Vetor(i, j, k);
}

double Vetor::angle(const Vetor &other) const {
    double cost = dot(other) / (norm() * other.norm());
    return acos(cost);
}

double Vetor::norm() const {
    return sqrt(x*x + y*y + z*z);
}

Vetor& Vetor::normalize() const {
    return Vetor(x/norm(), y/norm(), z/norm());
}

Vetor& Vetor::operator+(Vetor &other) const {
    return Vetor(x + other.x, y + other.y, z + other.z);
}

Vetor& Vetor::operator-(Vetor &other) const {
    return Vetor(x - other.x, y - other.y, z - other.z);
}

Vetor& Vetor::operator*(double p) const {
    return Vetor(x * p, y * p, z * p);
}

Vetor& Vetor::operator/(double p) const {
    return Vetor(x / p, y / p, z / p);
}
