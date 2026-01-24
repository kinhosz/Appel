#include <iostream>
#include <vector>
#include <Appel/geometry/point.h>
#include <Appel/geometry/triangle.h>
#include <Appel/entity/triangularMesh.h>
#include <Appel/entity/scene.h>
#include <Appel/graphic/camera.h>
#include <assert.h>
#include <Appel/graphic/utils.h>
using namespace std;
using namespace Appel;

#define WIDTH 640
#define HEIGHT 360

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
    tMesh.setPhongValues(0.8, 0.9, 0.1, 0.0, 0.0, 100.0);

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
    std::cerr << "clock_per_sec: " << t/CLOCKS_PER_SEC << "\n";
    assert(saveAsPng(frame, "assets/outputs/view/humanFace.png"));
}
