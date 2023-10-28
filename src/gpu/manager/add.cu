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
    host_cache->host_id[dvc_id] = gt.host_id;
    for(int j=0;j<3;j++) {
        host_cache->point[j].x[dvc_id] = gt.point[j].x;
        host_cache->point[j].y[dvc_id] = gt.point[j].y;
        host_cache->point[j].z[dvc_id] = gt.point[j].z;
    }

    return dvc_id;
}

#endif
