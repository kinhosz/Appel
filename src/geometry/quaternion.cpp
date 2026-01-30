#include <cmath>
#include <Appel/geometry/quaternion.h>

namespace Appel {
  Quaternion::Quaternion() {
    w = 1.0f;
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
  }

  Quaternion::Quaternion(double nw, double nx, double ny, double nz) {
    w = nw;
    x = nx;
    y = ny;
    z = nz;
  }

  Quaternion Quaternion::fromAxisAngle(double nx, double ny, double nz, double angle) {
    double halfAngle = angle * 0.5f;
    double s = std::sin(halfAngle);

    return Quaternion(
      std::cos(halfAngle),
      nx * s,
      ny * s,
      nz * s
    );
  }

  Quaternion Quaternion::fromAxisX(double angle) { return fromAxisAngle(1, 0, 0, angle); }
  Quaternion Quaternion::fromAxisY(double angle) { return fromAxisAngle(0, 1, 0, angle); }
  Quaternion Quaternion::fromAxisZ(double angle) { return fromAxisAngle(0, 0, 1, angle); }

  Quaternion Quaternion::operator*(const Quaternion& q) const {
    return Quaternion(
      w * q.w - x * q.x - y * q.y - z * q.z,
      w * q.x + x * q.w + y * q.z - z * q.y,
      w * q.y - x * q.z + y * q.w + z * q.x,
      w * q.z + x * q.y - y * q.x + z * q.w
    );
  }

  void Quaternion::normalize() {
    double mag = std::sqrt(w * w + x * x + y * y + z * z);
    if (mag > 0.0f) {
      double invMag = 1.0f / mag;
      w *= invMag; x *= invMag; y *= invMag; z *= invMag;
    }
  }

  Point Quaternion::rotatePoint(const Point& p) const {
    // q * p * !q
    double ix =  w * p.x + y * p.z - z * p.y;
    double iy =  w * p.y + z * p.x - x * p.z;
    double iz =  w * p.z + x * p.y - y * p.x;
    double iw = -x * p.x - y * p.y - z * p.z;

    return Point(
      ix * w + iw * -x + iy * -z - iz * -y,
      iy * w + iw * -y + iz * -x - ix * -z,
      iz * w + iw * -z + ix * -y - iy * -x
    );
  }
}
