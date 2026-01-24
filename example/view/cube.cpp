#include <iostream>
#include <vector>
#include <Appel/geometry/point.h>
#include <Appel/geometry/triangle.h>
#include <Appel/entity/triangularMesh.h>
#include <Appel/entity/scene.h>
#include <Appel/graphic/camera.h>
#include <assert.h>
#include <Appel/graphic/utils.h>
#include <SFML/Graphics.hpp>
#include <math.h>
using namespace std;
using namespace Appel;

#define WIDTH 640
#define HEIGHT 360
#define FPS 30

int main() {
  TriangularMesh cube("assets/models/cube.obj");
  cube.setTexture("assets/models/cube.png");

  Scene scene(1);
  scene.addObject(cube);
  double r = 5.0;
  double x, y;

  Camera camera(Point(5, 0, 0), Point(0, 0, 0), HEIGHT, WIDTH);

  Frame frame = camera.take(scene, false);
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Cube");

  double angle = 0.0;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) window.close();
    }

    window.clear();
    /************************************************/
    sf::Image image;
    image.create(WIDTH, HEIGHT, sf::Color::White);

    for (int x = 0; x < frame.horizontal(); ++x) {
      for (int y = 0; y < frame.vertical(); ++y) {
        sf::Color color(
          frame.getPixel(x, y).getRed(),
          frame.getPixel(x, y).getGreen(),
          frame.getPixel(x, y).getBlue()
        );
        image.setPixel(x, y, color);
      }
    }

    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite(texture);

    window.draw(sprite);

    frame = camera.take(scene, false);
    window.display();
  
    angle += (PI / (4.0 * FPS));
    x = cos(angle) * r;
    y = sin(angle) * r;

    camera.setPosition(Point(x, y, 0.0));
  }

  return 0;
}
