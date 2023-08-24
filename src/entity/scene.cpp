#include <entity/scene.h>
#include <geometry/utils.h>

Scene::Scene() {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size();
    this->environmentColor = Color(0, 0, 0);
    this->objectsCurrentIndex = 0;

    this->planes = std::map<int, Plane>();
}

Scene::Scene(const Color& environmentColor) {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size();
    this->environmentColor = environmentColor;
    this->objectsCurrentIndex = 0;
}

std::map<int, Light> Scene::getLights() const {
    return this->lights;
}

Color Scene::getEnvironmentColor() const {
    return this->environmentColor;
}

int Scene::addLight(const Light& light) {
    this->lights[lightsCurrentIndex] = light;
    this->lightsCurrentIndex++;
    return lightsCurrentIndex - 1;
}

void Scene::removeLight(int index) {
    assert(this->lights.find(index) != this->lights.end());
    this->lights.erase(index);
}

void Scene::setEnvironmentColor(const Color& environmentColor) {
    this->environmentColor = environmentColor;
}

int Scene::addObject(Plane object) {
    this->planes[objectsCurrentIndex] = object;
    this->objectsCurrentIndex++;
    return objectsCurrentIndex - 1;
}

int Scene::addObject(Sphere object) {
    this->spheres[objectsCurrentIndex] = object;
    this->objectsCurrentIndex++;
    return objectsCurrentIndex - 1;
}

int Scene::addObject(TriangularMesh object) {
    this->meshes[objectsCurrentIndex] = object;
    this->objectsCurrentIndex++;
    return objectsCurrentIndex - 1;
}

void Scene::removeObject(int index) {
    if(planes.find(index) != this->planes.end()) this->planes.erase(index);
    if(spheres.find(index) != this->spheres.end()) this->spheres.erase(index);
    if(meshes.find(index) != this->meshes.end()) this->meshes.erase(index);
}

SurfaceIntersection Scene::castRay(const Ray &ray) const {
    SurfaceIntersection nearSurface;

    for(const std::pair<int, Plane> tmp: planes) {
        const Plane plane = tmp.second;
        SurfaceIntersection current = plane.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface);
    }

    for(const std::pair<int, Sphere> tmp: spheres) {
        const Sphere sphere = tmp.second;
        SurfaceIntersection current = sphere.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface);
    }

    for(const std::pair<int, TriangularMesh> tmp: meshes) {
        const TriangularMesh mesh = tmp.second;
        SurfaceIntersection current = mesh.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface);
    }

    return nearSurface;
}

Color Scene::brightness(const Ray &ray, const SurfaceIntersection &surface) const {
    return Color(1, 1, 1);
}

Color Scene::traceRay(const Ray &ray) const {
    SurfaceIntersection surface = castRay(ray);

    return surface.color * brightness(ray, surface);
}
