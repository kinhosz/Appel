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
    int depth;

    Octree octree;
    Manager* manager;

    /* batch intersect stuffs */
    std::vector<Triangle> mappedTriangles;
    std::vector<std::pair<std::pair<double, double>, int>> sortedTrianglesIndexes;
    std::vector<std::pair<std::pair<double, double>, std::pair<double, int>>> activeIndexes;

    // TODO: Change Map to Vector
    std::map<int, Plane> planes;
    std::map<int, Sphere> spheres;
    std::map<int, TriangularMesh> meshes;

    std::vector<std::pair<int, int>> triangleIndex;
    std::vector<Triangle> triangles;

    Color brightness(const Ray& ray, SurfaceIntersection surface, const Box& box, const Light& light);
    Color phong(const Ray &ray, const SurfaceIntersection &surface, int index, int layer);

    /* batch intersect methods */
    void rebaseTriangles(const CoordinateSystem& cs);
    void sortTriangleIndexes();
    void activateTriangles(int& pointerToSortedIndexes, double planeSlopeVertical);
    SurfaceIntersection sweepOnTriangles(int& pointerToActives, double planeSlopeHorizontal, const Ray& ray);
    void deactivateTriangles(double planeSlopeVertical);

public:
    Scene(int depth=5);
    Scene(const Color& environmentColor, int depth=5);

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

    std::pair<SurfaceIntersection, int> castRay(const Ray &ray);
    Color traceRay(const Ray &ray, int layer);
    std::vector<std::vector<Color>> batchIntersect(const CoordinateSystem& cs, int width, int height, double distance);
};

#endif
