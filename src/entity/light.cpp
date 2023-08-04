#include <entity/light.h>
#include <assert.h>

Light::Light(Point location, Pixel intensity) {
    this->location = location;
    this->intensity = intensity;
}

Point Light::getLocation() const {
    return location;
}

Pixel Light::getIntensity() const {
    return intensity;
}

void Light::setLocation(Point location) {
    this->location = location;
}

void Light::setIntensity(Pixel intensity) {
    this->intensity = intensity;
}
