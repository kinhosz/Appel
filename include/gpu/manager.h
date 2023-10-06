#ifndef MANAGER_GPU_H
#define MANAGER_GPU_H

#include <set>
#include <map>
#include <vector>
#include <gpu/types/triangle.h>
#include <geometry/triangle.h>

class Manager {
    int calls_counter;
    int cache_limit;
    std::set<std::pair<int, int>> tableFrequency;
    std::map<int, int> hostToDeviceID;
    std::vector<std::pair<int, GTriangle>> lazy;
    GTriangle *cache;

    int *block_idx;
    float *block_dist;

    int *dvc_block_idx;
    float *dvc_block_dist;

    int threadsperblock;
    int blockspergrid;

    bool isOnCache(int host_id);
    int getFreeDeviceId();
    void pendingTransfer();

public:
    Manager();
    void transfer(int host_id, const Triangle &triangle);
    int run(const Ray &ray);

    ~Manager();
};

#endif
