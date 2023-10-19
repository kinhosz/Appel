#include <entity/triangularMesh.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <graphic/utils.h>
#include <geometry/utils.h>
using namespace std;

#define WIDTH 1920
#define HEIGHT 1080

TriangularMesh buildMesh() {
    Color red(1, 0, 0), green(0, 1, 0), blue(0, 0, 1), white(1, 1, 1);

    Point p0(100, 0, 0), p1(0, 100, 0), p2(-100, 0, 0), p3(0, -100, 0), p4(0, 0, 100);

    Vetor n1 = (Vetor(p1) - Vetor(p0)).cross(Vetor(p4) - Vetor(p0)).normalize();
    Vetor n2 = (Vetor(p2) - Vetor(p1)).cross(Vetor(p4) - Vetor(p1)).normalize();
    Vetor n3 = (Vetor(p3) - Vetor(p2)).cross(Vetor(p4) - Vetor(p2)).normalize();
    Vetor n4 = (Vetor(p0) - Vetor(p3)).cross(Vetor(p4) - Vetor(p3)).normalize();

    Vetor aux(0, 0, 1);

    Triangle t1(p0, p1, p4, aux, aux, aux, n1, red);
    Triangle t2(p1, p2, p4, aux, aux, aux, n2, green);
    Triangle t3(p2, p3, p4, aux, aux, aux, n3, blue);
    Triangle t4(p3, p0, p4, aux, aux, aux, n4, white);

    vector<Triangle> triangles;
    triangles.push_back(t1);
    triangles.push_back(t2);
    triangles.push_back(t3);
    triangles.push_back(t4);

    TriangularMesh tMesh(triangles, 0.60, 0.80, 0.20, 0.10, 0.10, 1.0);

    return tMesh;
}

Plane buildPlane() {
    Color purple(0.5, 0, 0.5);

    Point p0(100, 0, 0), p1(0, 100, 0), p2(-100, 0, 0), p3(0, -100, 0);

    Vetor n1 = (Vetor(p1) - Vetor(p0)).cross(Vetor(p2) - Vetor(p0)).normalize();

    Plane plane(p0, n1, purple, 0.60, 0.80, 0.20, 0.10, 0.10, 0.50);

    return plane;
}

Sphere buildSphere() {
    Color orange(1, 0.5, 0);

    Point center(200, 0, 50);
    double radius = 50;

    Sphere sphere(center, radius, orange, 0.60, 0.80, 0.20, 0.10, 0.00, 0.50);

    return sphere;
}

void image00(Scene &scene) {
    Camera camera(Point(-200, -200, 50), Point(0, 0, 50), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v1/version_01/image_00.png"));
}

void image01(Scene &scene) {
    Camera camera(Point(-200, -200, 200), Point(0, 0, 50), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v1/version_01/image_01.png"));
}

void image02(Scene &scene) {
    Camera camera(Point(-130, -200, 50), Point(0, 0, 100), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v1/version_01/image_02.png"));
}

void image03(Scene &scene) {
    Camera camera(Point(-100, 200, 100), Point(0, 0, 0), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v1/version_01/image_03.png"));
}

void image04(Scene &scene) {
    Camera camera(Point(300, 100, 300), Point(0, 0, 45), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v1/version_01/image_04.png"));
}

void image05(Scene &scene) {
    Camera camera(Point(200, 200, 300), Point(0, 0, 0), HEIGHT, WIDTH);
    Frame frame = camera.take(scene);
    assert(saveAsPng(frame, "assets/outputs/project_v1/version_01/image_05.png"));
}

int main() {
    Scene scene;
    scene.addObject(buildMesh());
    scene.addObject(buildPlane());
    scene.addObject(buildSphere());
    
    Light l2(Point(100, 10, 100), Color(1, 1, 1));

    scene.addLight(l2);

    image00(scene);
    image01(scene);
    image02(scene);
    image03(scene);
    image04(scene);
    image05(scene);
}
