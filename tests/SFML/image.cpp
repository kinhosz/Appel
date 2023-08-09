#include <SFML/Graphics.hpp>
#include <cassert>

int main() {
    sf::Image image;
    image.create(800, 600, sf::Color::Black); // Cria uma imagem preta de 800x600 pixels

    int x = 100;
    int y = 200;
    sf::Color color(255, 0, 0); // Cor vermelha (R=255, G=0, B=0)
    image.setPixel(x, y, color);

    if (image.saveToFile("bin/output.png")) {
        // Imagem salva com sucesso
    } else {
        // Ocorreu um erro ao salvar a imagem
        assert(false);
    }

    return 0;
}