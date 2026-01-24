#include <Appel/entity/light.h>
#include <assert.h>

namespace Appel {
    Light::Light() {
        this->location = Point();
        this->intensity = Color();
    }

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

    bool Light::operator==(const Light& other) const {
        return location == other.location && intensity == other.intensity;
    }

    bool Light::operator!=(const Light& other) const {
        return !(*this == other);
    }
}
