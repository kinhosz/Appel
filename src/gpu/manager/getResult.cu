#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>

int Manager::getResult(int id) const {
    return host_buffer[id];
}

#endif
