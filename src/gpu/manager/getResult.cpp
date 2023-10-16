#ifdef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <stdexcept>

int Manager::getResult(int id) const {
    throw std::runtime_error("GPU not found");
}

#endif
