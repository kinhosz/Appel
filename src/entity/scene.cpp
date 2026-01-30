#include <Appel/entity/scene.h>
#include <Appel/geometry/utils.h>
#include <Appel/geometry/coordinateSystem.h>
#include <memory>
#include <math.h>
#include <algorithm>

namespace Appel {
    Scene::Scene(int depth) : Scene(Color(), depth) {}

    Scene::Scene(const Color& environmentColor, int depth) {
        this->lights = std::map<int, Light>();
        this->lightsCurrentIndex = this->lights.size();
        this->environmentColor = environmentColor;
        this->objectsCurrentIndex = 0;

        this->triangleIndex = std::vector<std::pair<int, int>>();
        this->triangles = std::vector<Triangle>();

        double MIN_BORDER = -100000;
        double MAX_BORDER = 100000;

        this->depth = depth;

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

    Box* Scene::getObjectRef(int index) {
        if(planes.find(index) != planes.end()) return &planes.at(index);
        if(spheres.find(index) != this->spheres.end()) return &spheres.at(index);
        if(meshes.find(index) != this->meshes.end()) return &meshes.at(index);
        assert(false);
        return nullptr;
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
            const std::vector<int> indexes = octree.find(ray);
            for(int idx: indexes) {
                int triangle_id = triangleIndex[idx].second;
                const Triangle triangle = triangles[triangle_id];
                manager->add(triangle, idx);
            }

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

    void Scene::rebaseTriangles(const CoordinateSystem& cs) {
        mappedTriangles.clear();
        triangleToMesh.clear();
        for(const std::pair<const int, TriangularMesh>& p: meshes) {
            const Point &mesh_position = p.second.getPosition();
            const Quaternion &rotation = p.second.getRotation();

            const std::vector<Triangle>& triangles = p.second.getTriangles();
            for(const Triangle& triangle: triangles) {
                mappedTriangles.push_back(
                    triangle.rotate(
                        rotation
                    ).moveTo(
                        mesh_position
                    ).rebase(
                        cs
                    )
                );
                triangleToMesh.push_back(p.first);
            }
        }
    }

    void Scene::sortTriangleIndexes() {
        /* 
            Sorting triangles by the slope of z/y in the new coordinate system.
            All points with y < 0 are behind the observer and will not be rendered.
            The sortedTrianglesIndexes contains: <minSlope, maxSlope>, <index>
            This structure will be used for all rows of the image, from down to up
            simulating a sweep line.
        */
        sortedTrianglesIndexes.clear();
        for(int i=0;i<(int)mappedTriangles.size();i++) {
            const Triangle& triangle = mappedTriangles[i];
            double minSlope = 0.0, maxSlope = 0.0;
            bool hasSlope = false;

            for(int j=0;j<3;j++) {
                if(cmp(triangle.vertices[j].y, 0.0) <= 0) continue;

                double slope = triangle.vertices[j].z / triangle.vertices[j].y;

                if(!hasSlope) minSlope = slope, maxSlope = slope;
                minSlope = std::min(minSlope, slope);
                maxSlope = std::max(maxSlope, slope);
                hasSlope = true;
            }
            if(hasSlope) {
                if(cmp(minSlope, maxSlope) == 0) {
                    if(cmp(minSlope, 0.0) == -1) minSlope = -1;
                    else if(cmp(minSlope, 0.0) == 1) maxSlope = 1;
                }
                sortedTrianglesIndexes.push_back({{minSlope, maxSlope}, i});
            }
        }
        std::sort(sortedTrianglesIndexes.begin(), sortedTrianglesIndexes.end());
    }

    void Scene::activateTriangles(int& pointerToSortedIndexes, double planeSlopeVertical) {
        while(pointerToSortedIndexes < (int)sortedTrianglesIndexes.size()) {
            if(cmp(sortedTrianglesIndexes[pointerToSortedIndexes].first.first, planeSlopeVertical) == 1) break;
            double minSlopeH = 0.0, maxSlopeH = 0.0;
            int idx = sortedTrianglesIndexes[pointerToSortedIndexes].second;
            double maxSlopeV = sortedTrianglesIndexes[pointerToSortedIndexes].first.second;
            bool hasSlope = false;

            for(int i=0;i<3;i++){
                double slope = mappedTriangles[idx].vertices[i].x / mappedTriangles[idx].vertices[i].y;

                if(!hasSlope) {
                    minSlopeH = slope;
                    maxSlopeH = slope;
                    hasSlope = true;
                }
                minSlopeH = std::min(minSlopeH, slope);
                maxSlopeH = std::max(maxSlopeH, slope);
            }
            if(hasSlope) {
                if(cmp(minSlopeH, maxSlopeH) == 0) {
                    if(cmp(minSlopeH, 0.0) == -1) minSlopeH = -1;
                    else if(cmp(minSlopeH, 0.0) == 1) maxSlopeH = 1;
                }
                activeIndexes.push_back({{minSlopeH, maxSlopeH}, {maxSlopeV, idx}});
            }
            pointerToSortedIndexes++;
        }
        std::sort(activeIndexes.begin(), activeIndexes.end());
    }

    int Scene::sweepOnTriangles(int& pointerToActives, double planeSlopeHorizontal, const Ray& ray) {
        SurfaceIntersection nearSurface;
        int triangle_idx = -1;
        /*
            Passing for all triangles until the minSlopeH > planeSlopeHorizontal.
            However, if maxSlopeH < planeSlope, then deactivate it.
        */
        int current_active = pointerToActives;
        while(current_active < (int)activeIndexes.size()) {
            if(cmp(activeIndexes[current_active].first.first, planeSlopeHorizontal) == 1) break;
            if(cmp(activeIndexes[current_active].first.second, planeSlopeHorizontal) == -1) {
                std::swap(activeIndexes[current_active], activeIndexes[pointerToActives]);
                pointerToActives++;
                current_active++;
                continue;
            }
            /*
                Strong candidate: It matches on horizontal and vertical plane
            */
            int idx = activeIndexes[current_active].second.second;
            const Triangle& triangle = mappedTriangles[idx];

            SurfaceIntersection current = triangle.intersect(ray);
            if(current.distance < nearSurface.distance) {
                std::swap(current, nearSurface);
                std::swap(triangle_idx, idx);
            }
            current_active++;
        }
        return triangle_idx;
    }

    void Scene::deactivateTriangles(double planeSlopeVertical) {
        /*
            Now, after finish the sweep line for an entire row, we need clean up all triangles
            with maxSlopeV < planeSlopeVertical
        */
        for(int i=(int)activeIndexes.size()-1;i>=0;i--) {
            if(cmp(activeIndexes[i].second.first, planeSlopeVertical) < 1) {
                std::swap(activeIndexes[i], activeIndexes[(int)activeIndexes.size() - 1]);
                activeIndexes.pop_back();
            }
        }
    }

    std::vector<std::vector<Pixel>> Scene::batchIntersect(const CoordinateSystem& cs, int width, int height, double distance) {
        std::vector<std::vector<Pixel>> res(width, std::vector<Pixel>(height));

        rebaseTriangles(cs);
        sortTriangleIndexes();

        int pointerToSortedIndexes = 0;
        activeIndexes.clear();
        int mid_h = height / 2;

        for(int int_z=0;int_z<height;int_z++){
            double planeSlopeVertical = (double)(int_z - mid_h)/distance;

            activateTriangles(pointerToSortedIndexes, planeSlopeVertical);

            int mid_w = width / 2;
            int pointerToActives = 0;
            for(int int_x=0;int_x<width;int_x++){
                SurfaceIntersection nearSurface;
                int near_idx = -1;
                double planeSlopeHorizontal = (double)(int_x - mid_w)/distance;

                Vetor dir(planeSlopeHorizontal * distance, distance, planeSlopeVertical * distance);
                dir = dir.normalize();
                Ray ray(Point(0,0,0), dir);

                int curr_idx = sweepOnTriangles(pointerToActives, planeSlopeHorizontal, ray);
                SurfaceIntersection nearTriangle = (curr_idx == -1 ? SurfaceIntersection() : mappedTriangles[curr_idx].intersect(ray));
                if(cmp(nearTriangle.distance, nearSurface.distance) == -1) {
                    near_idx = curr_idx;
                    std::swap(nearSurface, nearTriangle);
                }

                if(near_idx != -1) {
                    int mesh_id = triangleToMesh[near_idx];
                    const TriangularMesh &mesh = meshes[mesh_id];
                    if(mesh.hasTexture()) {
                        Point p = ray.pointAt(nearSurface.distance);
                        std::pair<double, double> uv = mappedTriangles[near_idx].getUVAtPoint(p);
                        res[int_x][int_z] = mesh.getTexture(uv.first, uv.second);
                    } else {
                        res[int_x][int_z] = Pixel(mappedTriangles[near_idx].color);
                    }
                } else {
                    res[int_x][int_z] = Pixel(0, 0, 0);
                }
            }
            deactivateTriangles(planeSlopeVertical);
        }

        return res;
    }
}
