#ifndef MANAGER_GPU_H
#define MANAGER_GPU_H

#include <queue>
#include <gpu/types/ray.h>
#include <gpu/types/triangle.h>
#include <geometry/triangle.h>
#include <geometry/ray.h>

class Manager {
    int maxTriangles;
    int BATCHSIZE;
    std::queue<int> free_pos;

    GTriangle *cache;

    GRay *host_rays, *dvc_rays;

    int *rays_N, *triangles_N, *blocks_N;

    int threadsperblock_x;
    int threadsperblock_y;
    int bufferN;
    int *dvc_bufferN;

    int *buffer_idx;
    float *buffer_dist;

    int *host_res_idx, *dvc_res_idx;
public:
    Manager(int maxTriangles, int batchsize);
    ~Manager();

    int add(const Triangle& t, int host_id);
    std::vector<int> run(const std::vector<Ray> &partial);
};

#endif
