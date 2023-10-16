#include <gpu/manager.h>

int Manager::getLights() const {
    return this->free_light_pos;
}

int Manager::getDepth() const {
    return this->depth;
}
