#ifdef APPEL_GPU_DISABLED

#include <gpu/manager.h>
#include <stdexcept>

void Manager::run(const Vetor &up, const Vetor &right, 
    const Vetor &front, const Vetor &location, double dist) {

    throw std::runtime_error("GPU not found");
}

#endif
