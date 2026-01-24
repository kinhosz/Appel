#ifndef GRAPHIC_PIXEL_H
#define GRAPHIC_PIXEL_H

#include <Appel/graphic/color.h>

struct Pixel {
    int red, green, blue;

    Pixel();
    Pixel(Color color);
    Pixel(int red, int green, int blue);

    int getRed() const;
    int getGreen() const;
    int getBlue() const;

    void setRed(int red);
    void setGreen(int green);
    void setBlue(int blue);
};

#endif
