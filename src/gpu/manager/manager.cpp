#ifdef APPEL_GPU_DISABLED

#include <gpu/manager.h>

Manager::Manager() {
    this->hasAllocate = false;
}

Manager::Manager(unsigned int maxTriangles, int height, int width,
    int depth, int maxLights) {

    this->hasAllocate = false;
    this->height = height;
    this->width = width;
    this->depth = depth;
    this->maxLights = maxLights;

    this->free_light_pos = 0;

    this->triangles = maxTriangles;
    this->buffer_size = 0;
}

Manager::~Manager() {}

#endif
