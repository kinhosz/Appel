#ifndef ENTITY_SCENE_H
#define ENTITY_SCENE_H

#include <map>
#include <Appel/entity/light.h>
#include <Appel/entity/plane.h>
#include <Appel/entity/sphere.h>
#include <Appel/entity/triangularMesh.h>
#include <Appel/entity/box.h>
#include <Appel/graphic/color.h>
#include <Appel/datastructure/octree.h>
#include <Appel/gpu/manager.h>

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
    std::vector<int> triangleToMesh;
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
    int sweepOnTriangles(int& pointerToActives, double planeSlopeHorizontal, const Ray& ray);
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
    std::vector<std::vector<Pixel>> batchIntersect(const CoordinateSystem& cs, int width, int height, double distance);
};

#endif
