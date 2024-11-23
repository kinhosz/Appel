#ifndef ENTITY_BOX_H
#define ENTITY_BOX_H

#include <graphic/color.h>
#include <geometry/surfaceIntersection.h>
#include <geometry/ray.h>

class Box {
private:
    double diffuseCoefficient;
    double specularCoefficient;
    double ambientCoefficient;
    double reflectionCoefficient;
    double transmissionCoefficient;
    double roughnessCoefficient;
    double refractionIndex;
public:
    Box();
    virtual SurfaceIntersection intersect(const Ray& ray) const;

    double getDiffuseCoefficient() const;
    double getSpecularCoefficient() const;
    double getAmbientCoefficient() const;
    double getReflectionCoefficient() const;
    double getTransmissionCoefficient() const;
    double getRoughnessCoefficient() const;
    double getRefractionIndex() const;
    void setPhongValues(double kd, double ks, double ka, double kr, double kt, double roughness);
    void setRefractionIndex(double refractionIndex);
};

#endif
