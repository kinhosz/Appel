#include <Appel/datastructure/octreeNode.h>
#include <Appel/geometry/utils.h>

namespace Appel {
    OctreeNode::OctreeNode(double xl, double xr, double yl, double yr, double zl, double zr) {
        xL = xl;
        xR = xr;
        yL = yl;
        yR = yr;
        zL = zl;
        zR = zr;

        for(int i=0;i<8;i++) child[i] = -1;
    }

    double OctreeNode::getXL() const {
        return xL;        
    }

    double OctreeNode::getXR() const {
        return xR;
    }

    double OctreeNode::getYL() const {
        return yL;        
    }

    double OctreeNode::getYR() const {
        return yR;
    }

    double OctreeNode::getZL() const {
        return zL;        
    }

    double OctreeNode::getZR() const {
        return zR;
    }

    double OctreeNode::getMX() const {
        return (xL + xR)/2.0;
    }

    double OctreeNode::getMY() const {
        return (yL + yR)/2.0;
    }

    double OctreeNode::getMZ() const {
        return (zL + zR)/2.0;
    }

    void OctreeNode::add(int t_index) {
        surfaces.push_back(t_index);
    }

    int OctreeNode::getChild(int child_index) const {
        return child[child_index];
    }

    void OctreeNode::setChild(int child_index, int node_index) {
        child[child_index] = node_index;
    }

    std::vector<int> OctreeNode::getSurfaces() const {
        return surfaces;
    }
}
