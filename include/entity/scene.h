#ifndef ENTITY_SCENE_H
#define ENTITY_SCENE_H

#include <map>
#include <entity/light.h>
#include <entity/plane.h>
#include <entity/sphere.h>
#include <entity/triangularMesh.h>
#include <entity/box.h>
#include <graphic/color.h>
#include <datastructure/octree.h>
#include <gpu/manager.h>

#ifdef APPEL_GPU_DISABLED
#define ENABLE_GPU false
#else
#define ENABLE_GPU true
#endif

class Scene {
private:
    std::map<int, Light> lights;
    int lightsCurrentIndex;
    Color environmentColor;
    int objectsCurrentIndex;

    Octree octree;
    Manager manager;

    std::map<int, Plane> planes;
    std::map<int, Sphere> spheres;
    std::map<int, TriangularMesh> meshes;

    std::vector<std::pair<int, int>> triangleIndex;
    std::vector<Triangle> triangles;

    Color brightness(const Ray& ray, SurfaceIntersection surface, const Box& box, const Light& light, int cacheId);
    Color phong(const Ray &ray, const SurfaceIntersection &surface, int index, int layer, int offset, int node);

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

    std::pair<SurfaceIntersection, int> castRay(const Ray &ray, int cacheId);
    Color traceRay(const Ray &ray, int layer, int offset, int node);
    Manager& getManager();
};

#endif
