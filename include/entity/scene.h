#ifndef ENTITY_SCENE_H
#define ENTITY_SCENE_H

#include <map>
#include <entity/light.h>
#include <entity/plane.h>
#include <entity/sphere.h>
#include <entity/triangularMesh.h>
#include <entity/box.h>
#include <graphic/color.h>

class Scene {
private:
    std::map<int, Light> lights;
    int lightsCurrentIndex;
    Color environmentColor;
    int objectsCurrentIndex;

    std::map<int, Plane> planes;
    std::map<int, Sphere> spheres;
    std::map<int, TriangularMesh> meshes;

    Color brightness(const Ray& ray, SurfaceIntersection surface, const Box& box, const Light& light) const;
    Color phong(const Ray &ray, const SurfaceIntersection &surface, int index) const;

public:
    Scene();
    Scene(const Color& environmentColor);

    std::map<int, Light> getLights() const;
    Color getEnvironmentColor() const;

    int addLight(const Light& light);
    void removeLight(int index);
    void setEnvironmentColor(const Color& environmentColor);
    void removeObject(int index);

    int addObject(Plane object);
    int addObject(Sphere object);
    int addObject(TriangularMesh object);

    Box getObject(int index) const;

    std::pair<SurfaceIntersection, int> castRay(const Ray &ray) const;
    Color traceRay(const Ray &ray) const;
};

#endif
