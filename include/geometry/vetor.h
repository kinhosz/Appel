#ifndef GEOMETRY_VETOR_H
#define GEOMETRY_VETOR_H

struct Vetor{
private:
    void rotate(double &x, double &y, double alpha);
public:
    double x, y, z;
    Vetor();
    Vetor(double x, double y, double z);
    double dot(const Vetor &other) const;
    Vetor& cross(const Vetor &other) const;
    double angle(const Vetor &other) const;
    double norm() const;
    Vetor& normalize() const;
    Vetor& rotate_x(double alpha) const;
    Vetor& rotate_y(double alpha) const;
    Vetor& rotate_z(double alpha) const;

    Vetor& operator+(Vetor &other) const;
    Vetor& operator-(Vetor &other) const;
    Vetor& operator*(double p) const;
    Vetor& operator/(double p) const;
};

#endif
