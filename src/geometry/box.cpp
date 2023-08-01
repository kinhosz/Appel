#include <geometry/box.h>
#include <assert.h>

Box::Box() : kd(0), ks(0), ka(0), kr(0), kt(0), roughness(0), color(Color()) {}

Box::Box(double kd, double ks, double ka, double kr, double kt, double roughness, Color color)
    : kd(kd), ks(ks), ka(ka), kr(kr), kt(kt), roughness(roughness), color(color) {
    assert(kd >= 0 && kd <= 1);
    assert(ks >= 0 && ks <= 1);
    assert(ka >= 0 && ka <= 1);
    assert(kr >= 0 && kr <= 1);
    assert(kt >= 0 && kt <= 1);
    assert(roughness > 0);
}