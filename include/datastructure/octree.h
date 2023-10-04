#ifndef OCTREE_DATASTRUCTURE_H
#define OCTREE_DATASTRUCTURE_H

#include <vector>
#include <geometry/triangle.h>
#include <datastructure/octreeNode.h>
#include <geometry/utils.h>

class Octree {
    std::vector<OctreeNode> nodes;

    OctreeNode createNode(int current_node, int child_index) const;
    int add(const Triangle &triangle, int t_index, int current_node, int level);
    void find(const Ray &ray, double &current_t, int current_node, std::vector<int> &candidates) const;

public:
    Octree();
    Octree(double min_x, double max_x, double min_y, double max_y, double min_z, double max_z);

    int add(const Triangle &triangle, int t_index);
    std::vector<int> find(const Ray &ray) const;
};

#endif
