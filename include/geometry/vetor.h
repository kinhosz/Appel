#ifndef GEOMETRY_VETOR_H
#define GEOMETRY_VETOR_H

#include <geometry/point.h>

struct Vetor{
    double x, y, z;
    Vetor();
    Vetor(double x, double y, double z);
    Vetor(Point p);

    Point getPoint() const;
    double dot(const Vetor &other) const;
    Vetor cross(const Vetor &other) const;
    double angle(const Vetor &other) const;
    double norm() const;
    Vetor normalize() const;
    Vetor rotateX(double alpha) const;
    Vetor rotateY(double alpha) const;
    Vetor rotateZ(double alpha) const;

    Vetor operator+(const Point &other) const;
    Vetor operator+(const Vetor &other) const;
    Vetor operator-(const Vetor &other) const;
    Vetor operator*(double p) const;
    Vetor operator/(double p) const;
    bool operator==(const Vetor &other) const;
    bool operator!=(const Vetor &other) const;

    bool isOrthogonal(const Vetor &other) const;
    bool isParallel(const Vetor &other) const;
};

#endif
