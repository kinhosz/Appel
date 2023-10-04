#include <datastructure/octreeNode.h>
#include <geometry/utils.h>

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

bool OctreeNode::isInside(Point p) const {
    return (cmp(xL, p.x) <= 0 && cmp(p.x, xR) <= 0 &&
            cmp(yL, p.y) <= 0 && cmp(p.y, yR) <= 0 &&
            cmp(zL, p.z) <= 0 && cmp(p.z, zR) <= 0);
}

double OctreeNode::moveToNext(const Ray &ray, double t) const {
    Point p = ray.pointAt(t);

    double delta = DOUBLE_INF;
    double tmp;

    if(cmp(ray.direction.x, 0) != 0) {
        tmp = (xL - p.x) / ray.direction.x;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
        tmp = (getMX() - p.x) / ray.direction.x;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
        tmp = (xR - p.x) / ray.direction.x;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
    }

    if(cmp(ray.direction.y, 0) != 0) {
        tmp = (yL - p.y) / ray.direction.y;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
        tmp = (getMY() - p.y) / ray.direction.y;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
        tmp = (yR - p.y) / ray.direction.y;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
    }

    if(cmp(ray.direction.z, 0) != 0) {
        tmp = (zL - p.z) / ray.direction.z;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
        tmp = (getMZ() - p.z) / ray.direction.z;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
        tmp = (zR - p.z) / ray.direction.z;
        if(cmp(tmp, 0) != -1) delta = std::min(delta, tmp);
    }

    return delta + EPSILON;
}
