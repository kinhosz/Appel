#ifndef GRAPHIC_COLOR_H
#define GRAPHIC_COLOR_H

struct Color {
    enum ColorRepresentation { RGB, NORMALIZED_RGB };

    ColorRepresentation representation;
    union {
        struct {
            int red, green, blue;
        } rgb;
        struct {
            double red, green, blue;
        } normalized_rgb;
    };

    Color();
    Color(int red, int green, int blue, ColorRepresentation representation = RGB);
    Color(double normRed, double normGreen, double normBlue, ColorRepresentation representation = NORMALIZED_RGB);

    int getRed() const;
    int getGreen() const;
    int getBlue() const;
    double getNormRed() const;
    double getNormGreen() const;
    double getNormBlue() const;

    void setRed(int red);
    void setGreen(int green);
    void setBlue(int blue);
    void setNormRed(double normRed);
    void setNormGreen(double normGreen);
    void setNormBlue(double normBlue);
};

#endif