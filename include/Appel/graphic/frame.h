#ifndef GRAPHIC_FRAME_H
#define GRAPHIC_FRAME_H

#include <vector>
#include <Appel/graphic/pixel.h>

namespace Appel {
    class Frame {
    private:
        int verticalResolution;
        int horizontalResolution;
        std::vector<std::vector<Pixel>> matrix;

    public:
        Frame();
        Frame(int vResolution, int hResolution);
        Pixel getPixel(int x, int y) const;
        void setPixel(int x, int y, Pixel pixel);
        int vertical() const;
        int horizontal() const;
    };
}

#endif
