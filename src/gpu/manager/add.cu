#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <stdexcept>
#include <gpu/kernel.h>

int Manager::add(const Triangle& t, int host_id) {
    if(free_pos == maxTriangles) {
        throw std::runtime_error("Cache overflow!");
    }

    int dvc_id = free_pos++;

    GTriangle gt(t, host_id);
    tmp[dvc_id] = gt;

    return dvc_id;
}

#endif
