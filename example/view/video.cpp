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

#define WIDTH 640
#define HEIGHT 360
#define FPS 0.5

TriangularMesh buildHumanFace() {
    if(!freopen("assets/models/HumanFace.obj", "r", stdin)) {
        std::cerr << "error\n";
    }
    string tmp;
    int vertex_size;

    cin >> tmp;
    cin >> vertex_size;
    cin >> tmp;

    vector<Point> vertex;

    for(int i=0;i<vertex_size;i++) {
        cin >> tmp;
        double x, y, z;
        cin >> x >> y >> z;
        vertex.push_back(Point(x * 10000.0, y * 10000.0, z * 10000.0));
    }

    cin >> tmp;
    int faces;
    cin >> faces;
    cin >> tmp;

    vector<Triangle> triangles;

    Color c(0.85, 0.85, 0.85);

    for(int i=0;i<faces;i++) {
        cin >> tmp;
        int id1, id2, id3;
        cin >> id1 >> id2 >> id3;

        triangles.push_back({
            vertex[id1-1], vertex[id2-1], vertex[id3-1], c
        });
    }

    fclose(stdin);

    TriangularMesh tMesh(triangles);
    tMesh.setPhongValues(0.8, 0.9, 0.1, 0.0, 0.0, 100);

    return tMesh;
}

int main() {
    TriangularMesh humanFace = buildHumanFace();

    Scene scene;

    scene.addObject(humanFace);
    scene.addLight(Light(Point(0, 0, 10000), Color(1.0, 1.0, 1.0)));

    Camera camera(Point(0, -200, 1000), Point(0, 0, 0), HEIGHT, WIDTH);

    clock_t t;
    t = clock();
    Frame frame = camera.take(scene);
    t = clock() - t;

    double current_spent_time = (double)t/CLOCKS_PER_SEC;
    double desired_spent_time = 1.0/FPS;

    double angle = -3.14/2.0;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "video");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();

        /************************************************/
        t = clock();
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
        angle += 0.1;
        camera.setPosition(
            Point(
                200 * cos(angle),
                200 * sin(angle),
                1000
            )
        );

        double reduce_factor = min(1.0, desired_spent_time / current_spent_time);
        int reduced_width = sqrt(reduce_factor) * WIDTH;
        int reduced_height = sqrt(reduce_factor) * HEIGHT;

        camera.setResolution(reduced_width, reduced_height);
        frame = camera.take(scene);
        frame = camera.resize(frame, WIDTH, HEIGHT);

        t = clock() - t;
        int curr_fps = 1.0/((double)t/CLOCKS_PER_SEC);

        window.display();
        cerr << "FPS: " << curr_fps << "\n";
    }

    return 0;
}
