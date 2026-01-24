#include <Appel/entity/triangularMesh.h>
#include <Appel/entity/scene.h>
#include <Appel/graphic/camera.h>
#include <Appel/graphic/utils.h>
#include <Appel/geometry/utils.h>
#include <time.h>
#include <iostream>
using namespace std;
using namespace Appel;

#define WIDTH 1920
#define HEIGHT 1080

vector<TriangularMesh> buildMesh() {
    Point p0(0, 0, 0), p1(0, 500, 0), p2(300, 500, 0), p3(300, 0, 0),
    p4(0, 0, 250), p5(0, 500, 250), p6(300, 500, 250), p7(300, 0, 250);

    vector<TriangularMesh> meshes;

    vector<Triangle> triangles;

    Color c0(0.3, 0.4, 0.6), c1(0.8, 0.2, 0.4), c2(0.4, 0.2, 0.6), c3(0.7, 0.1, 0.6), c4(0.4, 0.7, 0.4);

    triangles.push_back(Triangle(
        p0, p3, p2, c0
    ));

    triangles.push_back(Triangle(
        p0, p2, p1, c0
    ));

    TriangularMesh tMesh1(triangles);
    tMesh1.setPhongValues(0.5, 0.9, 0.1, 0.1, 0.1, 1.0);

    meshes.push_back(tMesh1);

    triangles.clear();

    triangles.push_back(Triangle(
        p0, p4, p7, c1
    ));

    triangles.push_back(Triangle(
        p0, p7, p3, c1
    ));

    TriangularMesh tMesh2(triangles);
    tMesh2.setPhongValues(0.5, 0.5, 0.1, 0.1, 0.1, 1.0);

    meshes.push_back(tMesh2);

    triangles.clear();

    triangles.push_back(Triangle(
        p0, p1, p5, c2
    ));

    triangles.push_back(Triangle(
        p0, p5, p4, c2
    ));

    TriangularMesh tMesh3(triangles);
    tMesh3.setPhongValues(0.5, 0.5, 0.1, 0.1, 0.1, 1.0);

    meshes.push_back(tMesh3);

    triangles.clear();

    triangles.push_back(Triangle(
        p1, p2, p6, c3
    ));

    triangles.push_back(Triangle(
        p1, p6, p5, c3
    ));

    TriangularMesh tMesh4(triangles);
    tMesh4.setPhongValues(0.5, 0.5, 0.1, 0.1, 0.1, 1.0);

    meshes.push_back(tMesh4);

    triangles.clear();

    triangles.push_back(Triangle(
        p4, p5, p6, c4
    ));

    triangles.push_back(Triangle(
        p4, p6, p7, c4
    ));

    TriangularMesh tMesh5(triangles);
    tMesh5.setPhongValues(0.5, 0.5, 0.1, 0.1, 0.1, 100.0);

    meshes.push_back(tMesh5);

    triangles.clear();

    return meshes;
}

Sphere buildSphere0() {
    Color color(0.8, 0.4, 0.2);

    Point center(70, 60, 50);
    double radius = 50;

    Sphere sphere(center, radius, color);
    sphere.setPhongValues(0.80, 0.40, 0.00, 0.00, 0.00, 100.00);

    return sphere;
}

Sphere buildSphere1() {
    Color color(1.0, 1.0, 1.0);

    Point center(170, 200, 50);
    double radius = 50;

    Sphere sphere(center, radius, color);
    sphere.setPhongValues(0.80, 0.40, 0.10, 0.10, 1.00, 100.0);
    sphere.setRefractionIndex(10.0);

    return sphere;
}

Sphere buildSphere2() {
    Color color(1.0, 1.0, 1.0);

    Point center(70, 340, 50);
    double radius = 50;

    Sphere sphere(center, radius, color);
    sphere.setPhongValues(0.80, 0.40, 0.00, 1.00, 0.00, 100.0);

    return sphere;
}

Sphere buildSphere3() {
    Color color(1.0, 1.0, 1.0);

    Point center(170, 400, 10);
    double radius = 25;

    Sphere sphere(center, radius, color);
    sphere.setPhongValues(0.80, 0.40, 0.00, 1.00, 0.00, 100.0);

    return sphere;
}

void image00(Scene &scene) {
    Camera camera(Point(500, 250, 125), Point(0, 250, 125), HEIGHT, WIDTH);
    clock_t t;
    t = clock();
    Frame frame = camera.take(scene);
    t = clock() - t;
    std::cerr << "clock_per_sec: " << t/CLOCKS_PER_SEC << "\n";
    assert(saveAsPng(frame, "assets/outputs/project_v2/version_03/image_00.png"));
}

Plane buildPlane(){
    Plane plane(Point(0, 0, -10), Vetor(Point(0, 0, 1)), Color(1.0, 0.5, 0.25));
    plane.setPhongValues(0.5, 0.5, 0.1, 0.1, 0.1, 1.0);

    return plane;
}

int main() {
    Scene scene;
    vector<TriangularMesh> msh = buildMesh();

    scene.addObject(msh[0]);
    scene.addObject(msh[1]);
    scene.addObject(msh[2]);
    scene.addObject(msh[3]);
    scene.addObject(msh[4]);

    scene.addObject(buildSphere0());
    scene.addObject(buildSphere1());
    scene.addObject(buildSphere2());
    scene.addObject(buildSphere3());
    
    Light l(Point(120, 250, 200), Color(1.0, 1.0, 1.0));

    scene.addLight(l);

    scene.addObject(buildPlane());

    image00(scene);
}
