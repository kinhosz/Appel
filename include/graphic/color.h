#ifndef GRAPHIC_COLOR_H
#define GRAPHIC_COLOR_H

struct Color {
    int red, green, blue;
    
    Color();
    Color(int red, int green, int blue);

    int getRed() const;
    int getGreen() const;
    int getBlue() const;
    double getNormRed() const;
    double getNormGreen() const;
    double getNormBlue() const;

    void setRed(int red);
    void setGreen(int green);
    void setBlue(int blue);
};

#endif