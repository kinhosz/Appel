#ifndef MANAGER_CPU_H
#define MANAGER_CPU_H

#include <geometry/triangle.h>

class Manager {
public:
    Manager();
    void transfer(int host_id, const Triangle &triangle);
    int run(const Ray &ray);
};

#endif
