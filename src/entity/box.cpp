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
    this->refractionIndex = 1.0;
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
    this->refractionIndex = 1.0;

    this->velocity = Vetor(0, 0, 0);
}

SurfaceIntersection Box::intersect(const Ray& ray) const {
    return SurfaceIntersection();
}

double Box::getDiffuseCoefficient() const {
    return this->diffuseCoefficient;
}

double Box::getSpecularCoefficient() const {
    return this->specularCoefficient;
}

double Box::getAmbientCoefficient() const {
    return this->ambientCoefficient;
}

double Box::getReflectionCoefficient() const {
    return this->reflectionCoefficient;
}

double Box::getTransmissionCoefficient() const {
    return this->transmissionCoefficient;
}

double Box::getRoughnessCoefficient() const {
    return this->roughnessCoefficient;
}

double Box::getRefractionIndex() const {
    return this->refractionIndex;
}

void Box::setRefractionIndex(double refractionIndex) {
    this->refractionIndex = refractionIndex;
}

Vetor Box::getVelocity() const {
    return this->velocity;
}

void Box::setVelocity(Vetor velocity) {
    this->velocity = velocity;
}

bool Box::isMovable() const {
    return this->movable;
}

void Box::setMovable() {
    this->movable = true;
}
