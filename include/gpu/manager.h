#ifndef MANAGER_GPU_H
#define MANAGER_GPU_H

#include <queue>
#include <gpu/types/triangle.h>
#include <gpu/types/ray.h>
#include <geometry/triangle.h>
#include <geometry/vetor.h>

class Manager {
    GTriangle* cache_triangle;
    GPoint* cache_light;
    int* device_buffer;
    int* host_buffer;

    GRay* tmp_ray;

    unsigned int triangles;
    unsigned int buffer_size;

    int height, width;
    int depth;
    int free_light_pos;
    int maxLights;

    std::queue<int> free_pos;
public:
    Manager(unsigned int maxTriangles, int height, int width, 
        int depth, int maxLights);
    ~Manager();

    int addTriangle(const Triangle &triangle, int host_id);
    int addLight(const Point &p);
    void run(const Vetor &up, const Vetor &right, 
        const Vetor &front, const Vetor &location, double dist);
    int getResult(int id) const;
};

#endif
