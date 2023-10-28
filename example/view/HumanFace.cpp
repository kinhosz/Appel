#include <iostream>
#include <vector>
#include <geometry/point.h>
#include <geometry/triangle.h>
#include <entity/triangularMesh.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <assert.h>
#include <graphic/utils.h>
using namespace std;

#define WIDTH 640/4
#define HEIGHT 360/4

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

    return TriangularMesh(
        triangles,
        0.8, // kd
        0.9, // ks
        0.1, // ka
        0.0, // kr
        0.0, // kt
        100 // roughness
    );
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

/*
g++ -c -I include example/view/HumanFace.cpp -o bin/tmp.o -Wall -Werror -O2 -std=c++17
nvcc -Llib/SFML bin/tmp.o bin/obj/_graphic_frame.o bin/obj/_geometry_utils.o bin/obj/_entity_scene.o bin/obj/_graphic_utils.o bin/obj/_gpu_helper_multByScalar.o bin/obj/_gpu_helper_f_cmp.o bin/obj/_gpu_helper_dot.o bin/obj/_entity_plane.o bin/obj/_datastructure_graph.o bin/obj/_gpu_helper_triangleIntersect.o bin/obj/_gpu_helper_getDistance.o bin/obj/_gpu_helper_sub.o bin/obj/_gpu_manager_run.o bin/obj/_graphic_camera.o bin/obj/_gpu_helper_angle.o bin/obj/_gpu_manager_manager.o bin/obj/_graphic_color.o bin/obj/_graphic_pixel.o bin/obj/_gpu_types_point.o bin/obj/_gpu_helper_add.o bin/obj/_gpu_helper_norm.o bin/obj/_gpu_reducer_getMin.o bin/obj/_entity_light.o bin/obj/_geometry_point.o bin/obj/_gpu_helper_getNormal.o bin/obj/_datastructure_octree.o bin/obj/_gpu_manager_add.o bin/obj/_geometry_ray.o bin/obj/_gpu_helper_getReflection.o bin/obj/_geometry_vetor.o bin/obj/_gpu_helper_isInside.o bin/obj/_gpu_kernel_updateCache.o bin/obj/_gpu_helper_getRefraction.o bin/obj/_gpu_helper_normalize.o bin/obj/_geometry_surfaceIntersection.o bin/obj/_gpu_kernel_castRay.o bin/obj/_datastructure_octreeNode.o bin/obj/_entity_sphere.o bin/obj/_gpu_types_ray.o bin/obj/_gpu_helper_castRay.o bin/obj/_gpu_helper_cross.o bin/obj/_gpu_helper_pointAt.o bin/obj/_entity_box.o bin/obj/_entity_triangularMesh.o bin/obj/_geometry_triangle.o bin/obj/_gpu_types_triangle.o -o bin/example.exe
*/