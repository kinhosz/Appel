#include <entity/box.h>
#include <assert.h>
#include <geometry/ray.h>

Box::Box() {
    this->diffuseCoefficient = 0.0;
    this->specularCoefficient = 0.0;
    this->ambientCoefficient = 0.0;
    this->reflectionCoefficient = 0.0;
    this->transmissionCoefficient = 0.0;
    this->roughnessCoefficient = 0.0;
}

Box::Box(double kd, double ks, double ka, double kr, double kt, double roughness) {
    assert(kd >= 0.0 && kd <= 1.0);
    assert(ks >= 0.0 && ks <= 1.0);
    assert(ka >= 0.0 && ka <= 1.0);
    assert(kr >= 0.0 && kr <= 1.0);
    assert(kt >= 0.0 && kt <= 1.0);
    assert(roughness > 0.0);

    this->diffuseCoefficient = kd;
    this->specularCoefficient = ks;
    this->ambientCoefficient = ka;
    this->reflectionCoefficient = kr;
    this->transmissionCoefficient = kt;
    this->roughnessCoefficient = roughness;
}

SurfaceIntersection Box::intersect(Ray ray) const{
    return SurfaceIntersection();
}
