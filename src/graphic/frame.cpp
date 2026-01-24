#include <Appel/graphic/frame.h>
#include <cassert>

namespace Appel {
    Frame::Frame(): verticalResolution(0), horizontalResolution(0) {}

    Frame::Frame(int vResolution, int hResolution) {
        this->verticalResolution = vResolution;
        this->horizontalResolution = hResolution;

        matrix.resize(hResolution);
        for(int i=0;i<hResolution;i++){
            matrix[i].resize(vResolution, Pixel(0, 0, 0));
        }
    }

    Pixel Frame::getPixel(int x, int y) const {
        assert(x >= 0 && x < this->horizontalResolution);
        assert(y >= 0 && y < this->verticalResolution);

        return matrix[x][y];
    }

    void Frame::setPixel(int x, int y, Pixel pixel) {
        assert(x >= 0 && x < this->horizontalResolution);
        assert(y >= 0 && y < this->verticalResolution);

        this->matrix[x][y] = pixel;
    }

    int Frame::vertical() const {
        return this->verticalResolution;
    }
    int Frame::horizontal() const {
        return this->horizontalResolution;
    }
}
