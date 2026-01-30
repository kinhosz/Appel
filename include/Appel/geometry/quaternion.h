#ifndef QUATERNION_H
#define QUATERNION_H

#include <cmath>
#include <Appel/geometry/vetor.h>

namespace Appel {
  class Quaternion {
  public:
    double w, x, y, z;

    Quaternion();
    Quaternion(double nw, double nx, double ny, double nz);

    static Quaternion fromAxisAngle(double nx, double ny, double nz, double angle);

    static Quaternion fromAxisX(double angle);
    static Quaternion fromAxisY(double angle);
    static Quaternion fromAxisZ(double angle);

    Quaternion operator*(const Quaternion& q) const;

    void normalize();
    Point rotatePoint(const Point& p) const;
  };
}

#endif
