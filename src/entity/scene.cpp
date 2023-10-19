#include <entity/scene.h>
#include <geometry/utils.h>
#include <memory>
#include <math.h>

Scene::Scene() : Scene(Color()) {}

Scene::Scene(const Color& environmentColor) {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size();
    this->environmentColor = environmentColor;
    this->objectsCurrentIndex = 0;

    this->triangleIndex = std::vector<std::pair<int, int>>();
    this->triangles = std::vector<Triangle>();

    double MIN_BORDER = -100000;
    double MAX_BORDER = 100000;

    this->depth = 5;

    this->octree = Octree(MIN_BORDER, MAX_BORDER, MIN_BORDER, MAX_BORDER, MIN_BORDER, MAX_BORDER);
    this->manager = new Manager(50000);
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

    const std::vector<Triangle> meshTriangles = object.getTriangles();

    for(Triangle triangle: meshTriangles) {
        int node = octree.add(triangle, triangles.size());
        assert(node != -1);
        manager->add(triangle, triangleIndex.size());
        triangleIndex.push_back(std::make_pair(objectsCurrentIndex-1, triangles.size()));

        triangles.push_back(triangle);
    }

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

std::pair<SurfaceIntersection, int> Scene::castRay(const Ray &ray) {
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

    if(ENABLE_GPU) {
        int idx = manager->run(ray);

        if(idx != -1) {
            int object_id = triangleIndex[idx].first;
            int triangle_id = triangleIndex[idx].second;

            const Triangle triangle = triangles[triangle_id];

            SurfaceIntersection current = triangle.getSurface(ray);
            if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = object_id;
        }
    }
    else {
        const std::vector<int> indexes = octree.find(ray);
        for(int idx: indexes) {
            int object_id = triangleIndex[idx].first;
            int triangle_id = triangleIndex[idx].second;

            const Triangle triangle = triangles[triangle_id];

            SurfaceIntersection current = triangle.intersect(ray);
            if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = object_id;
        }
    }

    return std::make_pair(nearSurface, index);
}

Color Scene::brightness(const Ray& ray, SurfaceIntersection surface, const Box& box, const Light& light) {
    if(cmp(ray.direction.angle(surface.normal * -1.0), PI/2.0) == 1) surface.normal = surface.normal * -1.0;

    Point matched = ray.pointAt(surface.distance);
    Vetor dir = (Vetor(light.getLocation()) - Vetor(matched)).normalize();

    Ray temp(matched, dir);

    double delta = 0.01;

    Ray lightRay(temp.pointAt(delta), dir);

    std::pair<SurfaceIntersection, int> opaqueSurface = castRay(lightRay);

    Vetor toLight = Vetor(Vetor(light.getLocation()) -Vetor(lightRay.location));

    if(cmp(opaqueSurface.first.distance, toLight.norm()) == -1) return Color(0, 0, 0);
    Color color = light.getIntensity();

    if(cmp(lightRay.direction.angle(surface.normal), PI/2.0) != -1) return Color(0, 0, 0);

    Vetor lightReflex = surface.getReflection(lightRay.direction);

    double diffuse = box.getDiffuseCoefficient() * (lightRay.direction.dot(surface.normal));
    double specular = box.getSpecularCoefficient() * std::pow(lightReflex.dot(ray.direction * -1.0), box.getRoughnessCoefficient());

    color = color * (diffuse + specular);

    return color;
}

Color Scene::phong(const Ray &ray, const SurfaceIntersection &surface, int index, int layer) {
    if(index == -1) return Color(0, 0, 0);
    if(layer >= this->depth) return Color(0, 0, 0);

    const Box box = getObject(index);

    Color color(0, 0, 0);
    color = color + (environmentColor * box.getAmbientCoefficient());

    for(std::pair<int, Light> tmp: lights) {
        const Light light = tmp.second;

        color = color + brightness(ray, surface, box, light);
    }

    color = color + (
        traceRay(Ray(ray.pointAt(surface.distance - 0.01), surface.getReflection(ray.direction * -1.0)), layer+1)
            * box.getReflectionCoefficient()
    );

    color = color + (
        traceRay(Ray(ray.pointAt(surface.distance + 0.01), surface.getRefraction(ray.direction * -1.0, box.getRefractionIndex())), layer+1)
            * box.getTransmissionCoefficient()
    );

    return color;
}

Color Scene::traceRay(const Ray &ray, int layer) {
    std::pair<SurfaceIntersection, int> match = castRay(ray);
    SurfaceIntersection surface = match.first;
    int index = match.second;

    return surface.color * phong(ray, surface, index, layer);
}
