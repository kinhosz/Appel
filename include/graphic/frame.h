#ifndef GRAPHIC_FRAME_H
#define GRAPHIC_FRAME_H

#include <vector>
#include <graphic/color.h>

class Frame {
private:
    int verticalResolution;
    int horizontalResolution;
    std::vector<std::vector<Color>> matrix;

public:
    Frame();
    Frame(int vResolution, int hResolution);
    Color getPixel(int x, int y) const;
    void setPixel(int x, int y, Color color);
    int vertical() const;
    int horizontal() const;
};

#endif
