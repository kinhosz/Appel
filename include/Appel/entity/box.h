#ifndef ENTITY_BOX_H
#define ENTITY_BOX_H

#include <Appel/graphic/color.h>
#include <Appel/geometry/surfaceIntersection.h>
#include <Appel/geometry/ray.h>

namespace Appel {
    class Box {
    private:
        /* Phong */
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

        /* Phong */
        double getDiffuseCoefficient() const;
        double getSpecularCoefficient() const;
        double getAmbientCoefficient() const;
        double getReflectionCoefficient() const;
        double getTransmissionCoefficient() const;
        double getRoughnessCoefficient() const;
        double getRefractionIndex() const;
        void setPhongValues(double kd, double ks, double ka, double kr, double kt, double roughness);
        void setRefractionIndex(double refractionIndex);

        /* Entity */
        virtual Point getPosition() const;
        virtual void moveTo(const Point &p);

        virtual double getXRotation() const;
        virtual double getYRotation() const;
        virtual double getZRotation() const;

        virtual void setXRotation(double alpha);
        virtual void setYRotation(double alpha);
        virtual void setZRotation(double alpha);
    };
}

#endif
