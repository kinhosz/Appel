#include <geometry/size.h>
#include <geometry/utils.h>
#include <cassert>

Size::Size() : width(0), height(0) {}

Size::Size(int width, int height) {
    assert(width >= 0 && height >= 0);

    this->width = width;
    this->height = height;
}

void Size::setWidth(int width) {
    assert(width >= 0);

    this->width = width;
}

void Size::setHeight(int height) {
    assert(height >= 0);

    this->height = height;
}

bool Size::operator==(const Size& other) const {
    return cmp(width, other.width) == 0 && cmp(height, other.height) == 0;
}

bool Size::operator!=(const Size& other) const {
    return !(*this == other);
}
