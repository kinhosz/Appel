#include <datastructure/octree.h>
#include <math.h>
#include <geometry/utils.h>

Octree::Octree() {}

Octree::Octree(double min_x, double max_x, double min_y, double max_y, double min_z, double max_z) {
    nodes.push_back(OctreeNode(min_x, max_x, min_y, max_y, min_z, max_z));
}

OctreeNode Octree::createNode(int current_node, int child_index) const {
    double xl, xr, yl, yr, zl, zr;

    if((child_index&(1)) == 0) xl = nodes[current_node].getXL(), xr = nodes[current_node].getMX();
    else xl = nodes[current_node].getMX(), xr = nodes[current_node].getXR();

    if((child_index&(2)) == 0) yl = nodes[current_node].getYL(), yr = nodes[current_node].getMY();
    else yl = nodes[current_node].getMY(), yr = nodes[current_node].getYR();

    if((child_index&(4)) == 0) zl = nodes[current_node].getZL(), zr = nodes[current_node].getMZ();
    else zl = nodes[current_node].getMZ(), zr = nodes[current_node].getZR();

    return OctreeNode(xl, xr, yl, yr, zl, zr);
}

int Octree::add(const Triangle &triangle, int t_index, int current_node, int level) {
    bool addHere = false;
    int child_index = 0;

    Point minT = triangle.getMin();
    Point maxT = triangle.getMax();

    double mx = nodes[current_node].getMX();
    double my = nodes[current_node].getMY();
    double mz = nodes[current_node].getMZ();

    if((cmp(minT.x, mx) <= 0 && cmp(maxT.x, mx) >= 0) ||
        (cmp(minT.y, my) <= 0 && cmp(maxT.y, my) >= 0) ||
        (cmp(minT.z, mz) <= 0 && cmp(maxT.z, mz) >= 0)) {
            addHere = true;
    }

    if(addHere) {
        nodes[current_node].add(t_index);
        return current_node;
    }

    if(cmp(minT.x, mx) == 1) child_index += 1;
    if(cmp(minT.y, my) == 1) child_index += 2;
    if(cmp(minT.z, mz) == 1) child_index += 4;

    int next_node = nodes[current_node].getChild(child_index);

    if(next_node == -1) {
        next_node = nodes.size();
        OctreeNode tmp_node = createNode(current_node, child_index);
        nodes.push_back(tmp_node);
        nodes[current_node].setChild(child_index, next_node);
    }

    return add(triangle, t_index, next_node, level+1);
}

bool Octree::isInside(const Ray &ray, int current_node) const {
    double t0 = 0.0;
    double tf = DOUBLE_INF;

    double tl, tr;

    if(cmp(ray.direction.x, 0) != 0){
        tl = (nodes[current_node].getXL() - ray.location.x) / ray.direction.x;
        tr = (nodes[current_node].getXR() - ray.location.x) / ray.direction.x;
        if(cmp(tl, tr) == 1) std::swap(tl, tr);

        t0 = std::max(t0, tl);
        tf = std::min(tf, tr);
    }

    if(cmp(ray.direction.y, 0) != 0){
        tl = (nodes[current_node].getYL() - ray.location.y) / ray.direction.y;
        tr = (nodes[current_node].getYR() - ray.location.y) / ray.direction.y;
        if(cmp(tl, tr) == 1) std::swap(tl, tr);

        t0 = std::max(t0, tl);
        tf = std::min(tf, tr);
    }

    if(cmp(ray.direction.z, 0) != 0){
        tl = (nodes[current_node].getZL() - ray.location.z) / ray.direction.z;
        tr = (nodes[current_node].getZR() - ray.location.z) / ray.direction.z;
        if(cmp(tl, tr) == 1) std::swap(tl, tr);

        t0 = std::max(t0, tl);
        tf = std::min(tf, tr);
    }

    return cmp(t0, tf) == -1;
}

void Octree::find(const Ray &ray, int current_node, std::vector<int> &candidates) const {
    if(current_node == -1) return;
    if(!isInside(ray, current_node)) return;

    const std::vector<int> tmp = nodes[current_node].getSurfaces();
    for(int idx: tmp) candidates.push_back(idx);

    for(int i=0;i<8;i++) find(ray, nodes[current_node].getChild(i), candidates);
}

int Octree::add(const Triangle &triangle, int t_index) {
    return add(triangle, t_index, 0, 0);
}

std::vector<int> Octree::find(const Ray &ray) const {
    std::vector<int> candidates;
    find(ray, 0, candidates);
    return candidates;
}
