#ifndef OCTREENODE_DATASTRUCTURE_H
#define OCTREENODE_DATASTRUCTURE_H

#include <vector>
#include <geometry/point.h>
#include <geometry/ray.h>

class OctreeNode {
    double xL, xR;
    double yL, yR;
    double zL, zR;
    int child[8];
    std::vector<int> surfaces;

public:
    OctreeNode(double xl, double xr, double yl, double yr, double zl, double zr);

    double getXL() const;
    double getXR() const;
    double getYL() const;
    double getYR() const;
    double getZL() const;
    double getZR() const;
    double getMX() const;
    double getMY() const;
    double getMZ() const;

    void add(int t_index);
    int getChild(int child_index) const;
    void setChild(int child_index, int node_index);

    std::vector<int> getSurfaces() const;
};

#endif
