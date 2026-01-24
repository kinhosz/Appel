#ifndef APPEL_GPU_DISABLED

#include <Appel/gpu/manager.h>
#include <stdexcept>
#include <Appel/gpu/kernel.h>

namespace Appel {
    int Manager::add(const Triangle& t, int host_id) {
        if(free_pos == (int)maxTriangles) {
            throw std::runtime_error("Cache overflow!");
        }

        int dvc_id = free_pos++;

        GTriangle gt(t, host_id);
        tmp[dvc_id] = gt;

        return dvc_id;
    }
}

#endif
