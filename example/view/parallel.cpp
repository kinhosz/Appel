#include <vector>
#include <geometry/point.h>
#include <geometry/triangle.h>
#include <entity/triangularMesh.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <SFML/Graphics.hpp>
#include <math.h>
using namespace std;

#define WIDTH 640
#define HEIGHT 360
#define FPS 30

TriangularMesh buildTriangles() {
    vector<Triangle> ts;
    // ground
    ts.push_back(Triangle(Point(0, 100, 0), Point(100, 0, 0), Point(-100, 0, 0), Color(0.5, 0.39, 0.83)));
    // front
    ts.push_back(Triangle(Point(-100, 0, 0), Point(100, 0, 0), Point(0, 0, 100), Color(0.78, 0.65, 0.98)));
    // left
    ts.push_back(Triangle(Point(-100, 0, 0), Point(-100, 100, 0), Point(-100, 50, 100), Color(0.87, 0.24, 0.48)));
    // right
    ts.push_back(Triangle(Point(100, 0, 0), Point(100, 100, 0), Point(100, 50, 100), Color(0.48, 0.24, 0.87)));

    return TriangularMesh(
        ts,
        0.8, // kd
        0.9, // ks
        0.1, // ka
        0.0, // kr
        0.0, // kt
        100 // roughness
    );
}

int main() {
    TriangularMesh mesh = buildTriangles();

    Scene scene(1);
    scene.addObject(mesh);

    double y = 1000;
    Camera camera(Point(0, 1000, 20), Point(0, 0, 20), HEIGHT, WIDTH);

    Frame frame = camera.take(scene, false);

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "video");

    clock_t t2;

    while (window.isOpen()) {
        t2 = clock();
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
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
        /************************************************/

        y -= (y * 0.01);
        camera.setPosition(Point(0, y, 20));
        frame = camera.take(scene, false);

        window.display();

        double rs = (clock() - t2) * 1000.0 / CLOCKS_PER_SEC;
        while(rs < (1000.0 / FPS)) {
            rs = (clock() - t2) * 1000.0 / CLOCKS_PER_SEC;
        }
    }

    return 0;
}
