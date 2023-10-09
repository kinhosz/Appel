#ifdef APPEL_GPU_DISABLED

#include <cpu/manager.h>
#include <stdexcept>

Manager::Manager() {}

void Manager::transfer(int host_id, const Triangle &triangle) {
    throw std::runtime_error("This method can only be used with the GPU enabled.");
}

int Manager::run(const Ray &ray) {
    throw std::runtime_error("This method can only be used with the GPU enabled.");
}

#endif
