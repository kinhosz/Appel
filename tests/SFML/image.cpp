#include <SFML/Graphics.hpp>
#include <cassert>

int main() {
    sf::Image image;
    image.create(800, 600, sf::Color::Black);

    int x = 100;
    int y = 200;
    sf::Color color(255, 0, 0);
    image.setPixel(x, y, color);

    assert(image.saveToFile("bin/output.png"));

    return 0;
}
