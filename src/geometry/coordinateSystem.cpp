#include <Appel/geometry/coordinateSystem.h>
#include <Appel/geometry/utils.h>
#include <cmath>

namespace Appel {
    CoordinateSystem::CoordinateSystem() {
        this->origin = Point(0, 0, 0);
        this->ux = Vetor(1, 0, 0);
        this->uy = Vetor(0, 1, 0);
        this->uz = Vetor(0, 0, 1);
    }

    CoordinateSystem::CoordinateSystem(Point origin, Vetor ux, Vetor uy, Vetor uz) {
        this->origin = origin;

        this->ux = ux.normalize(); 
        this->uy = uy.normalize();
        this->uz = uz.normalize();
    }

    Point CoordinateSystem::rebase(const Point& p) const {
        Vetor v = Vetor(p) - Vetor(origin);

        double x_new = v.dot(this->ux);
        double y_new = v.dot(this->uy);
        double z_new = v.dot(this->uz);

        return Point(x_new, y_new, z_new);
    }
}
