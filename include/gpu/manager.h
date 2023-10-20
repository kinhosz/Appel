#ifndef MANAGER_GPU_H
#define MANAGER_GPU_H

#include <queue>
#include <gpu/types/ray.h>
#include <gpu/types/triangle.h>
#include <geometry/triangle.h>
#include <geometry/ray.h>

class Manager {
    unsigned int maxTriangles;

    GTriangle* cache;
    int* buffer_idx;
    float* buffer_dist;
    int* result;

    GRay* dvc_ray;
    int* dvc_N;
    int* dvc_BLOCK;

    int threadsPerBlock;
    int blocksPerGrid;

    int free_pos;
    GTriangle* tmp;
public:
    Manager(unsigned int maxTriangles);
    ~Manager();

    int add(const Triangle& t, int host_id);
    int run(const Ray& ray);
};

#endif
