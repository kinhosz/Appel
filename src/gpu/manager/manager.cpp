#ifdef APPEL_GPU_DISABLED

#include <gpu/manager.h>

Manager::Manager(int maxTriangles, int batchsize) {}
Manager::~Manager() {}

int Manager::add(const Triangle& t, int host_id) {
    return -1;
}

std::vector<int> Manager::run(const std::vector<Ray> &rays) {
    return std::vector<int>();
}

#endif
