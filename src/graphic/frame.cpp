#include <graphic/frame.h>
#include <cassert>

Frame::Frame(): verticalResolution(0), horizontalResolution(0) {}

Frame::Frame(int vResolution, int hResolution) {
    this->verticalResolution = vResolution;
    this->horizontalResolution = hResolution;

    matrix.resize(vResolution);
    for(int i=0;i<vResolution;i++){
        matrix[i].resize(hResolution, Pixel(0, 0, 0));
    }
}

Pixel Frame::getPixel(int x, int y) const {
    assert(x >= 0 && x < this->verticalResolution);
    assert(y >= 0 && y < this->horizontalResolution);

    return matrix[x][y];
}

void Frame::setPixel(int x, int y, Pixel pixel) {
    assert(x >= 0 && x < this->verticalResolution);
    assert(y >= 0 && y < this->horizontalResolution);

    this->matrix[x][y] = pixel;
}

int Frame::vertical() const {
    return this->verticalResolution;
}
int Frame::horizontal() const {
    return this->horizontalResolution;
}
