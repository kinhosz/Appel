#ifndef GEOMETRY_VETOR_H
#define GEOMETRY_VETOR_H

struct Vetor{
public:
    double x, y, z;
    Vetor();
    Vetor(double x, double y, double z);
    double dot(const Vetor &other) const;
    Vetor& cross(const Vetor &other) const;
    double angle(const Vetor &other) const;
    double norm() const;
    Vetor& normalize() const;
    Vetor& rotate_x() const;
    Vetor& rotate_y() const;
    Vetor& rotate_z() const;

    Vetor& operator+(Vetor &other) const;
    Vetor& operator-(Vetor &other) const;
    Vetor& operator*(double p) const;
    Vetor& operator/(double p) const;
};

#endif