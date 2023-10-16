#ifdef APPEL_GPU_DISABLED

#include <gpu/manager.h>

int Manager::addTriangle(const Triangle &triangle, int host_id) {
    return -1;
}

#endif
