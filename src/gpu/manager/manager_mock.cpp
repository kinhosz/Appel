#ifdef APPEL_GPU_DISABLED

#include <Appel/gpu/manager.h>

namespace Appel {
    Manager::Manager(unsigned int maxTriangles) {}
    Manager::~Manager() {}

    int Manager::add(const Triangle& t, int host_id) {
        return -1;
    }

    int Manager::run(const Ray& ray) {
        return -1;
    }
}

#endif
