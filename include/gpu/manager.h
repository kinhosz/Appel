#ifndef MANAGER_GPU_H
#define MANAGER_GPU_H

#include <queue>
#include <gpu/types/ray.h>
#include <gpu/types/triangleArray.h>
#include <geometry/triangle.h>
#include <geometry/ray.h>

class Manager {
    int maxTriangles;
    int BATCHSIZE;
    int free_pos;

    GTriangleArray *cache, *host_cache, *aux;

    GRay *host_rays, *dvc_rays;

    int *rays_N, *triangles_N, *blocks_N;

    int threadsperblock_x;
    int threadsperblock_y;

    float *dvc_res_dist, *host_res_dist;

    int *dvc_buffer_idx, *host_buffer_idx;
    float *dvc_buffer_dist, *host_buffer_dist;

    int minClock, maxClock, cntCalls;
    float avgClock;
public:
    Manager(int maxTriangles, int batchsize);
    ~Manager();

    int add(const Triangle& t, int host_id);
    std::vector<int> run(const std::vector<Ray> &partial);
};

#endif
