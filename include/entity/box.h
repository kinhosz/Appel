#ifndef ENTITY_BOX_H
#define ENTITY_BOX_H

#include <graphic/color.h>
#include <geometry/surfaceIntersection.h>
#include <geometry/ray.h>
#include <geometry/vetor.h>

class Box {
private:
    double diffuseCoefficient;
    double specularCoefficient;
    double ambientCoefficient;
    double reflectionCoefficient;
    double transmissionCoefficient;
    double roughnessCoefficient;
    double refractionIndex;
    Vetor velocity;
    bool movable = false;
public:
    Box();
    Box(double kd, double ks, double ka, double kr, double kt, double roughness);
    virtual SurfaceIntersection intersect(const Ray& ray) const;

    double getDiffuseCoefficient() const;
    double getSpecularCoefficient() const;
    double getAmbientCoefficient() const;
    double getReflectionCoefficient() const;
    double getTransmissionCoefficient() const;
    double getRoughnessCoefficient() const;
    double getRefractionIndex() const;
    void setRefractionIndex(double refractionIndex);
    Vetor getVelocity() const;
    void setVelocity(Vetor velocity);
    bool isMovable() const;
    void setMovable();
};

#endif
