#ifndef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <stdexcept>
#include <gpu/kernel.h>

void Manager::addLight(const Point &p) {
    if(this->free_light_pos == this->maxLights) {
        throw std::runtime_error("The GPU cache has reached its maximum capacity");
    }

    updateCacheLight<<<1,1>>>(this->free_light_pos, p, cache_light);
    this->free_light_pos++;
}

#endif
