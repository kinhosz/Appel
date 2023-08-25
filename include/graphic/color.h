#ifndef GRAPHIC_COLOR_H
#define GRAPHIC_COLOR_H

struct Color {
    double red, green, blue;

    Color();
    Color(double red, double green, double blue);

    double getRed() const;
    double getGreen() const;
    double getBlue() const;

    void setRed(double red);
    void setGreen(double green);
    void setBlue(double blue);

    bool operator==(const Color& other) const;
    bool operator!=(const Color& other) const;
    
    Color operator*(const Color& other) const;
    Color operator+(const Color& other) const;

    Color operator*(double k) const;

    double truncate(double c) const;
};

#endif