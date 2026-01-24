#ifndef OCTREE_DATASTRUCTURE_H
#define OCTREE_DATASTRUCTURE_H

#include <vector>
#include <Appel/geometry/triangle.h>
#include <Appel/datastructure/octreeNode.h>
#include <Appel/geometry/utils.h>

namespace Appel {
    class Octree {
        std::vector<OctreeNode> nodes;

        OctreeNode createNode(int current_node, int child_index) const;
        int add(const Triangle &triangle, int t_index, int current_node, int level);
        void find(const Ray &ray, int current_node, std::vector<int> &candidates) const;
        bool isInside(const Ray &ray, int current_node) const;

    public:
        Octree();
        Octree(double min_x, double max_x, double min_y, double max_y, double min_z, double max_z);

        int add(const Triangle &triangle, int t_index);
        std::vector<int> find(const Ray &ray) const;
    };
}

#endif
