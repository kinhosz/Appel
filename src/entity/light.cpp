#include <light.h>
#include <assert.h>

Light::Light(Point location, Color intensity) {
    this->location = location;
    this->intensity = intensity;
}

Point Light::getLocation() const {
    return location;
}

Color Light::getIntensity() const {
    return intensity;
}

void Light::setLocation(Point location) {
    this->location = location;
}

void Light::setIntensity(Color intensity) {
    this->intensity = intensity;
}
