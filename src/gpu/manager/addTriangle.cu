#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <gpu/kernel.h>
#include <stdexcept>

int Manager::addTriangle(const Triangle &triangle, int host_id) {
    if(free_pos.empty()) {
        throw std::runtime_error("The GPU cache has reached its maximum capacity");
    }

    int dvc_id = free_pos.front();
    free_pos.pop();

    updateCacheTriangle<<<1,1>>>(dvc_id, GTriangle(triangle, host_id), cache_triangle);
    return dvc_id;
}

#endif
