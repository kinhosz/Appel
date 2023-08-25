#include <entity/scene.h>
#include <geometry/utils.h>
#include <memory>
#include <math.h>

Scene::Scene() {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size();
    this->environmentColor = Color(0, 0, 0);
    this->objectsCurrentIndex = 0;
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
    if(planes.find(index) != planes.end()) this->planes.erase(index);
    if(spheres.find(index) != spheres.end()) this->spheres.erase(index);
    if(meshes.find(index) != meshes.end()) this->meshes.erase(index);
}

Box Scene::getObject(int index) const {
    if(planes.find(index) != planes.end()) return planes.at(index);
    if(spheres.find(index) != this->spheres.end()) return spheres.at(index);
    if(meshes.find(index) != this->meshes.end()) return meshes.at(index);
    assert(false);
}

std::pair<SurfaceIntersection, int> Scene::castRay(const Ray &ray) const {
    SurfaceIntersection nearSurface;
    int index = -1;

    for(const std::pair<int, Plane> tmp: planes) {
        const Plane plane = tmp.second;
        SurfaceIntersection current = plane.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = tmp.first;
    }

    for(const std::pair<int, Sphere> tmp: spheres) {
        const Sphere sphere = tmp.second;
        SurfaceIntersection current = sphere.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = tmp.first;
    }

    for(const std::pair<int, TriangularMesh> tmp: meshes) {
        const TriangularMesh mesh = tmp.second;
        SurfaceIntersection current = mesh.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = tmp.first;
    }

    return std::make_pair(nearSurface, index);
}

Color Scene::brightness(const Ray& ray, SurfaceIntersection surface, const Box& box, const Light& light) const {
    if(cmp(ray.direction.angle(surface.normal * -1.0), PI/2.0) == 1) surface.normal = surface.normal * -1.0;

    Point matched = ray.pointAt(surface.distance);
    Vetor dir = (Vetor(light.getLocation()) - Vetor(matched)).normalize();
    Ray lightRay(matched, dir);

    std::pair<SurfaceIntersection, int> opaqueSurface = castRay(lightRay);
    if(cmp(opaqueSurface.first.distance, surface.distance) != 1) return Color(0, 0, 0);

    Color color = light.getIntensity();

    // todo: analysing observer pov
    double diffuse = box.getDiffuseCoefficient() * (lightRay.direction.dot(surface.normal));
    double specular = box.getSpecularCoefficient() * std::pow(lightRay.direction.dot(surface.normal), box.getRoughnessCoefficient());

    color = color * (diffuse + specular);

    return color;
}

Color Scene::phong(const Ray &ray, const SurfaceIntersection &surface, int index) const {
    if(index == -1) return Color(1, 1, 1);

    const Box box = getObject(index);

    Color color(0, 0, 0);
    color = color + (environmentColor * box.getAmbientCoefficient());

    for(std::pair<int, Light> tmp: lights) {
        const Light light = tmp.second;

        color = color + brightness(ray, surface, box, light);
    }

    return color;
}

Color Scene::traceRay(const Ray &ray) const {
    std::pair<SurfaceIntersection, int> match = castRay(ray);
    SurfaceIntersection surface = match.first;
    int index = match.second;

    return surface.color * phong(ray, surface, index);
}
