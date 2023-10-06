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

    bool isOnCache(int host_id);
    int getFreeDeviceId();
    void pendingTransfer();

public:
    Manager();
    void transfer(int host_id, const Triangle &triangle);
    int run(const Ray &ray);
};

#endif
