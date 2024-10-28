#include <iostream>
#include <vector>
#include <geometry/point.h>
#include <geometry/triangle.h>
#include <entity/triangularMesh.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <assert.h>
#include <graphic/utils.h>
#include <SFML/Graphics.hpp>
#include <math.h>
using namespace std;

#define WIDTH 640
#define HEIGHT 360
#define FPS 30

TriangularMesh buildTriangles() {
    vector<Triangle> ts;
    int y = 200;

    vector<Color> colors;
    colors.push_back(Color(1, 0, 0));
    colors.push_back(Color(0, 1, 0));
    colors.push_back(Color(0, 0, 1));
    colors.push_back(Color(1, 1, 0));
    colors.push_back(Color(1, 0, 1));
    colors.push_back(Color(0, 1, 1));

    for(int x=-100;x<100;x++) {
        for(int z=-100;z<100;z++) {
            ts.push_back(Triangle(Point(x, y, z), Point(x+1, y, z), Point(x, y, z+1), colors[random()%6]));
            ts.push_back(Triangle(Point(x+1, y, z), Point(x+1, y, z+1), Point(x, y, z+1), colors[random()%6]));
        }
    }

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

    double focus_x = 0.0;
    double focus_y = 0.0;
    double focus_z = 0.0;
    double r = 10.0;
    Camera camera(Point(0, 0, 0), Point(0, 10000.0, 0), HEIGHT, WIDTH);
    double pos_y = 199.0;

    Frame frame = camera.take(scene, false);

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "video");

    clock_t t2;
    double angle = 0.0;

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
        if(cmp(pos_y, 0.0) > 0) {
            pos_y -= (200.0 - pos_y) * 0.01;
            if(cmp(pos_y, 0.0) <= 0) pos_y = 0.0;
            camera.setPosition(Point(0, pos_y, 0));
        }
        else{
            focus_y = cos(angle) * r;
            focus_x = sin(angle) * r;
            camera.setFocus(Point(focus_x, focus_y, focus_z));
            angle += (PI / (4.0 * FPS));
        }

        frame = camera.take(scene, false);

        window.display();

        double rs = (clock() - t2) * 1000.0 / CLOCKS_PER_SEC;
        while(rs < (1000.0 / FPS)) {
            rs = (clock() - t2) * 1000.0 / CLOCKS_PER_SEC;
        }

        double elapsed = (double)(clock() - t2) * 1000.0/CLOCKS_PER_SEC;
        cerr << "-------------------\n";
        cerr << "FPS: " << (int)1.0/(elapsed/1000.0) << ", time: " << elapsed << "ms\n";
    }

    return 0;
}
