#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <stdexcept>
#include <gpu/kernel.h>

int Manager::add(const Triangle& t, int host_id) {
    if(free_pos.empty()) {
        throw std::runtime_error("Cache overflow!");
    }

    int dvc_id = free_pos.front();
    free_pos.pop();

    GTriangle gt(t, host_id);
    updateCache<<<1,1>>>(dvc_id, gt, cache);

    return dvc_id;
}

#endif
