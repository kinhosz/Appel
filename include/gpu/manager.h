#ifndef MANAGER_GPU_H
#define MANAGER_GPU_H

#include <queue>
#include <gpu/types/ray.h>
#include <gpu/types/triangle.h>
#include <geometry/triangle.h>
#include <geometry/ray.h>

class Manager {
    std::queue<int> free_pos;
    unsigned int maxTriangles;

    GTriangle* cache;
    int* buffer;

    GRay* dvc_ray;
    int* dvc_N;
public:
    Manager(unsigned int maxTriangles);
    ~Manager();

    int add(const Triangle& t, int host_id);
    int run(const Ray& ray);
};

#endif
