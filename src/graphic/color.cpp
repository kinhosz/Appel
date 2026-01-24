#include <Appel/graphic/color.h>
#include <Appel/geometry/utils.h>

namespace Appel {
    Color::Color() {
        red = 0.0;
        green = 0.0;
        blue = 0.0;
    }

    Color::Color(double red, double green, double blue) {
        red = truncate(red);
        green = truncate(green);
        blue = truncate(blue);

        this->red = red;
        this->green = green;
        this->blue = blue;
    }

    double Color::truncate(double c) const {
        c = std::min(c, 1.0);
        c = std::max(c, 0.0);

        return c;
    }

    double Color::getRed() const {
        return red;
    }

    double Color::getGreen() const {
        return green;
    }

    double Color::getBlue() const {
        return blue;
    }

    void Color::setRed(double red) {
        assert(red >= 0.0 && red <= 1.0);

        this->red = red;
    }

    void Color::setGreen(double green) {
        assert(green >= 0.0 && green <= 1.0);

        this->green = green;
    }

    void Color::setBlue(double blue) {
        assert(blue >= 0.0 && blue <= 1.0);

        this->blue = blue;
    }

    bool Color::operator==(const Color& other) const {
        return cmp(red, other.red) == 0 && cmp(green, other.green) == 0 && cmp(blue, other.blue) == 0;
    }

    bool Color::operator!=(const Color& other) const {
        return !(*this == other);
    }

    Color Color::operator*(const Color &other) const {
        return Color(red * other.red, green * other.green, blue * other.blue);
    }

    Color Color::operator+(const Color &other) const {
        return Color(red + other.red, green + other.green, blue + other.blue);
    }

    Color Color::operator*(double k) const {
        return Color(red * k, green * k, blue * k);
    }
}
