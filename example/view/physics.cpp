#include <iostream>
#include <geometry/point.h>
#include <geometry/vetor.h>
#include <entity/sphere.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <SFML/Graphics.hpp>
#include <math.h>
using namespace std;

#define WIDTH 640
#define HEIGHT 380
#define FPS 15

Plane buildPlane() {
    Point p(0, 0, 0);
    Vetor n(0, 0, 1);
    Color red(1, 0, 0);

    return Plane(p, n, red, 0.60, 0.80, 0.20, 0.80, 0.10, 1.00);
}

Sphere buildSphere() {
    Color green(0, 1, 0);
    Point center(0, -100, 100);
    double radius = 10;

    Sphere sphere(center, radius, green, 0.60, 0.80, 0.20, 0.80, 0.00, 10.0);
    sphere.setMovable();
    sphere.setVelocity(Vetor(0, 50, 0));

    return sphere;
}

int main() {
    Scene scene;

    scene.setAcceleration(Vetor(0, 0, -20));

    scene.addObject(buildPlane());
    scene.addObject(buildSphere());
    scene.addLight(Light(Point(200, 80, 2000), Color(1.0, 1.0, 1.0)));
    scene.addLight(Light(Point(200, 50, 10), Color(1.0, 1.0, 1.0)));

    Camera camera(Point(200, 50, 50), Point(0, 50, 50), HEIGHT, WIDTH);

    clock_t t;
    t = clock();
    scene.simulate(0.0);
    Frame frame = camera.take(scene);

    double desired_spent_time = 1.0/FPS;

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "physics");

    while (window.isOpen()) {
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
        t = clock() - t;
        double time_elapsed_in_sec = (double)t/CLOCKS_PER_SEC;
        t = clock();
        scene.simulate(time_elapsed_in_sec);

        double reduce_factor = min(1.0, desired_spent_time / time_elapsed_in_sec);
        int reduced_width = sqrt(reduce_factor) * WIDTH;
        int reduced_height = sqrt(reduce_factor) * HEIGHT;

        if(desired_spent_time < time_elapsed_in_sec && 1 == 2) camera.setResolution(reduced_width, reduced_height);
        frame = camera.take(scene);
        frame = camera.resize(frame, WIDTH, HEIGHT);

        int curr_fps = 1.0/time_elapsed_in_sec;

        window.display();
        cerr << "FPS: " << curr_fps << "\n";
    }

    return 0;
}
