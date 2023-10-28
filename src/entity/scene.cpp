#include <entity/scene.h>
#include <geometry/utils.h>
#include <memory>
#include <math.h>
#include <datastructure/graph.h>
#include <iostream>
#include <set>

Scene::Scene() : Scene(Color()) {}

Scene::Scene(const Color& environmentColor) {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size();
    this->environmentColor = environmentColor;
    this->objectsCurrentIndex = 0;

    this->triangleIndex = std::vector<std::pair<int, int>>();
    this->triangles = std::vector<Triangle>();

    double MIN_BORDER = -100000000;
    double MAX_BORDER = 1000000000;

    this->depth = 5;
    this->batchsize = 1;

    this->castRayTable.resize(batchsize);

    this->octree = Octree(MIN_BORDER, MAX_BORDER, MIN_BORDER, MAX_BORDER, MIN_BORDER, MAX_BORDER);

    int maxTriangles = 64 * 1024;

    this->manager = new Manager(maxTriangles, batchsize);
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

std::pair<SurfaceIntersection, int> Scene::intersectOnPlanes(const Ray &ray) const {
    SurfaceIntersection nearSurface;
    int index = -1;

    for(const std::pair<int, Plane> tmp: planes) {
        const Plane plane = tmp.second;
        SurfaceIntersection current = plane.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = tmp.first;
    }

    return std::make_pair(nearSurface, index);
}

std::pair<SurfaceIntersection, int> Scene::intersectOnSpheres(const Ray &ray) const {
    SurfaceIntersection nearSurface;
    int index = -1;

    for(const std::pair<int, Sphere> tmp: spheres) {
        const Sphere sphere = tmp.second;
        SurfaceIntersection current = sphere.intersect(ray);
        if(current.distance < nearSurface.distance) std::swap(current, nearSurface), index = tmp.first;
    }

    return std::make_pair(nearSurface, index);
}

std::pair<SurfaceIntersection, int> Scene::castRay(const Ray &ray) {
    SurfaceIntersection nearSurface;
    int index = -1;

    std::pair<SurfaceIntersection, int> candidate = intersectOnPlanes(ray);
    if(cmp(nearSurface.distance, candidate.first.distance) == 1) std::swap(candidate.first, nearSurface), index = candidate.second;

    candidate = intersectOnSpheres(ray);
    if(cmp(nearSurface.distance, candidate.first.distance) == 1) std::swap(candidate.first, nearSurface), index = candidate.second;

    if(ENABLE_GPU) {
        int idx = castRayTable[currentBatch].front();
        castRayTable[currentBatch].pop();

        if(idx != -1) {
            int object_id = triangleIndex[idx].first;
            int triangle_id = triangleIndex[idx].second;

            const Triangle triangle = triangles[triangle_id];

            SurfaceIntersection current = triangle.getSurface(ray);
            if(cmp(current.distance, nearSurface.distance) == -1) std::swap(current, nearSurface), index = object_id;
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

int Scene::getBatchSize() const {
    return batchsize;
}

void Scene::traceRayInBatch(const std::vector<Ray> &rays, std::vector<Color> &result) {
    std::queue<std::pair<int, Ray>> lazy;
    std::queue<int> lazy_levels;
    std::vector<Graph> graphs(rays.size());
    std::queue<int> parent;

    for(int i=0;i<(int)rays.size();i++) {
        lazy.push(std::make_pair(i, rays[i]));
        lazy_levels.push(0);
        parent.push(0);
    }

    while(!lazy.empty()) {
        std::vector<Ray> partial;
        std::vector<int> batch_ids;
        std::vector<int> partial_level;
        std::vector<int> partial_parent;

        for(int i=0;i<batchsize && !lazy.empty();i++) {
            batch_ids.push_back(lazy.front().first);
            partial.push_back(lazy.front().second);
            if(isnan(lazy.front().second.direction.x)) {
                assert(false);
            }
            lazy.pop();

            partial_level.push_back(lazy_levels.front());
            lazy_levels.pop();

            partial_parent.push_back(parent.front());
            parent.pop();
        }

        std::set<int> candidates;

        for(int i=0;i<(int)partial.size();i++) {
            const std::vector<int> cand = octree.find(partial[i]);
            for(int j=0;j<(int)cand.size();j++) {
                candidates.insert(cand[j]);
            }
        }

        if(candidates.size() > 10000) std::cerr << candidates.size() << "\n";

        for(int c: candidates) {
            int triangle_id = triangleIndex[c].second;
            const Triangle triangle = triangles[triangle_id];

            manager->add(triangle, c);
        }

        const std::vector<int> host_ids = manager->run(partial);

        for(int i=0;i<(int)batch_ids.size();i++) {
            SurfaceIntersection near;
            SurfaceIntersection curr = intersectOnPlanes(partial[i]).first;
            if(cmp(near.distance, curr.distance) == 1) near = curr;

            curr = intersectOnSpheres(partial[i]).first;
            if(cmp(near.distance, curr.distance) == 1) near = curr;

            int host_id = host_ids[i];

            if(host_id != -1) {
                int t_id = triangleIndex[host_id].second;
                const Triangle triangle = triangles[t_id];
                
                curr = triangle.getSurface(partial[i]);

                if(cmp(curr.distance, near.distance) >= 0) host_id = -1;
                else near = curr;
            }

            int vertex = graphs[batch_ids[i]].addEdge(partial_parent[i], host_id);

            if(host_id == -1) continue;
            if(partial_level[i] == -1) continue;
            if(partial_level[i] == this->depth) continue;

            const Ray ray = partial[i];
            const Box box = getObject(triangleIndex[host_id].first);

            for(std::pair<int, Light> tmp: lights) {
                const Light light = tmp.second;

                Point match = ray.pointAt(near.distance);

                Ray lightRay(match, (Vetor(light.getLocation()) - Vetor(match)).normalize());
                lightRay.location = lightRay.pointAt(0.01);

                if(isnan(lightRay.direction.x)) {
                    std::cerr << light.getLocation().x << ", " << light.getLocation().y << ", " << light.getLocation().z << "\n";
                    std::cerr << match.x << ", " << match.y << ", " << match.z << "\n";
                    std::cerr << near.distance << "<- distance\n";
                    std::cerr << host_id << "<- host\n";
                    assert(false);
                }

                lazy.push(std::make_pair(batch_ids[i], lightRay));
                lazy_levels.push(-1);

                parent.push(vertex);
            }

            Ray reflexRay(ray.pointAt(near.distance - 0.01), near.getReflection(ray.direction * -1.0));
            Ray refractRay(ray.pointAt(near.distance + 0.01), near.getRefraction(ray.direction * -1.0, box.getRefractionIndex()));

            if(isnan(reflexRay.direction.x)) {
                assert(false);
            }
            if(isnan(refractRay.direction.x)) {
                assert(false);
            }

            lazy.push(std::make_pair(batch_ids[i], reflexRay));
            lazy_levels.push(partial_level[i] + 1);
            parent.push(vertex);

            lazy.push(std::make_pair(batch_ids[i], refractRay));
            lazy_levels.push(partial_level[i] + 1);
            parent.push(vertex);
        }
    }

    currentBatch = 0;
    while(currentBatch < (int)rays.size()) {
        std::vector<int> path;
        graphs[currentBatch].dfs(0, path);

        for(int i=0;i<(int)path.size(); i++) {
            castRayTable[currentBatch].push(path[i]);
        }

        Color color = traceRay(rays[currentBatch], 0);
        result[currentBatch] = color;

        currentBatch++;
    }
}
