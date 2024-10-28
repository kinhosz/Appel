#include <geometry/coordinateSystem.h>
#include <geometry/utils.h>
#include <cmath>

CoordinateSystem::CoordinateSystem() {}

CoordinateSystem::CoordinateSystem(Point origin, Vetor ux, Vetor uy, Vetor uz) {
    this->origin = origin;

    this->angle_z = getAngle(ux.x, ux.y);
    ux = ux.rotateZ(-angle_z);
    uy = uy.rotateZ(-angle_z);
    uz = uz.rotateZ(-angle_z);

    this->angle_y = getAngle(ux.x, ux.z);
    ux = ux.rotateY(-angle_y);
    uy = uy.rotateY(-angle_y);
    uz = uz.rotateY(-angle_y);

    this->angle_x = getAngle(uy.y, uy.z);
}

Point CoordinateSystem::rebase(const Point& p) const {
    Vetor v = Vetor(p) - Vetor(origin);
    v = v.rotateZ(-angle_z);
    v = v.rotateY(-angle_y);
    v = v.rotateX(-angle_x);
    return Point(v.x, v.y, v.z);
}
