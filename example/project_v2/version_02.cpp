#include <entity/triangularMesh.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <graphic/utils.h>
#include <geometry/utils.h>
using namespace std;

#define WIDTH 1920
#define HEIGHT 1080

vector<TriangularMesh> buildMesh() {
    Color white(1.0, 1.0, 1.0);

    Point p0(0, 0, 0), p1(700, 0, 0), p2(700, 700, 0), p3(0, 700, 0);
    Point p4(0, 0, 700), p5(700, 0, 700), p6(700, 700, 700), p7(0, 700, 700);

    Triangle t1(p0, p1, p2, white);
    Triangle t2(p0, p2, p3, white);

    Triangle t3(p0, p3, p4, white);
    Triangle t4(p3, p7, p4, white);

    Triangle t5(p0, p4, p5, white);
    Triangle t6(p0, p5, p1, white);

    Triangle t7(p1, p5, p6, white);
    Triangle t8(p1, p6, p2, white);

    Triangle t9(p2, p6, p7, white);
    Triangle t10(p2, p7, p3, white);

    Triangle t11(p4, p6, p5, white);
    Triangle t12(p4, p7, p6, white);

    vector<Triangle> triangles;

    triangles.push_back(t1);
    triangles.push_back(t2);
    triangles.push_back(t5);
    triangles.push_back(t6);
    triangles.push_back(t7);
    triangles.push_back(t8);
    triangles.push_back(t9);
    triangles.push_back(t10);
    triangles.push_back(t11);
    triangles.push_back(t12);

    TriangularMesh tMesh(triangles, 0.10, 0.10, 0.20, 0.70, 0.10, 10.0);

    triangles.clear();

    triangles.push_back(t3);
    triangles.push_back(t4);

    TriangularMesh tMesh2(triangles, 0.10, 0.10, 0.20, 0.70, 0.00, 200.0);

    vector<TriangularMesh> meshes;

    meshes.push_back(tMesh);
    meshes.push_back(tMesh2);

    return meshes;
}

Sphere buildSphere0() {
    Color color(1.0, 1.0, 1.0);

    Point center(100, 100, 100);
    double radius = 100;

    Sphere sphere(center, radius, color, 0.30, 0.30, 0.20, 0.80, 0.00, 1.0);

    return sphere;
}

Sphere buildSphere1() {
    Color color(0.6, 0.4, 0.3);

    Point center(100, 400, 100);
    double radius = 100;

    Sphere sphere(center, radius, color, 0.60, 0.80, 0.20, 0.80, 0.00, 5.00);

    return sphere;
}

Sphere buildSphere2() {
    Color color(0.2, 0.7, 0.6);

    Point center(400, 100, 100);
    double radius = 100;

    Sphere sphere(center, radius, color, 0.70, 0.20, 0.20, 0.00, 0.00, 20.00);

    return sphere;
}

Sphere buildSphere3() {
    Color color(0.25, 0.4, 0.5);

    Point center(400, 400, 100);
    double radius = 100;

    Sphere sphere(center, radius, color, 0.60, 0.30, 0.20, 0.80, 0.00, 1.0);

    return sphere;
}

void image00(Scene &scene) {
    Camera camera(Point(590, 10, 590), Point(0, 500, 0), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v2/version_02/image_00.png"));
}

void image01(Scene &scene) {
    Camera camera(Point(590, 590, 590), Point(0, 0, 0), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v2/version_02/image_01.png"));
}

void image02(Scene &scene) {
    Camera camera(Point(590, 350, 500), Point(0, 250, 0), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v2/version_02/image_02.png"));
}

void image03(Scene &scene) {
    Camera camera(Point(590, 350, 500), Point(0, 250, 150), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v2/version_02/image_03.png"));
}

int main() {
    Scene scene;
    vector<TriangularMesh> msh = buildMesh();

    scene.addObject(msh[0]);
    scene.addObject(msh[1]);
    scene.addObject(buildSphere0());
    scene.addObject(buildSphere1());
    scene.addObject(buildSphere2());
    scene.addObject(buildSphere3());
    
    Light l(Point(490, 10, 490), Color(1.0, 1.0, 1.0));

    scene.addLight(l);

    image00(scene);
    image01(scene);
    image02(scene);
    image03(scene);
}
