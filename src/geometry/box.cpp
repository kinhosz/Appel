#include <geometry/box.h>
#include <assert.h>

Box::Box() {
    this->diffuse_coefficient = 0.0;
    this->specular_coefficient = 0.0;
    this->ambient_coefficient = 0.0;
    this->reflection_coefficient = 0.0;
    this->transmission_coefficient = 0.0;
    this->roughness = 0.0;
}

Box::Box(double kd, double ks, double ka, double kr, double kt, double roughness) {
    assert(kd >= 0.0 && kd <= 1.0);
    assert(ks >= 0.0 && ks <= 1.0);
    assert(ka >= 0.0 && ka <= 1.0);
    assert(kr >= 0.0 && kr <= 1.0);
    assert(kt >= 0.0 && kt <= 1.0);
    assert(roughness > 0.0);

    this->diffuse_coefficient = kd;
    this->specular_coefficient = ks;
    this->ambient_coefficient = ka;
    this->reflection_coefficient = kr;
    this->transmission_coefficient = kt;
    this->roughness = roughness;
}