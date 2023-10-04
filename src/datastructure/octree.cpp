#include <datastructure/octree.h>

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

int Octree::add(const Triangle &triangle, int t_index, int current_node) {
    bool addHere = false;
    int child_index = 0;

    Point minT = triangle.getMin();
    Point maxT = triangle.getMax();

    double mx = nodes[current_node].getMX();
    double my = nodes[current_node].getMY();
    double mz = nodes[current_node].getMZ();

    if(cmp(minT.x, mx) == -1 && cmp(maxT.x, mx) == 1 ||
        cmp(minT.y, my) == -1 && cmp(maxT.y, my) == 1 ||
        cmp(minT.z, mz) == -1 && cmp(maxT.z, mz) == 1) {
            addHere = true;
    }

    if(addHere) {
        nodes[current_node].add(t_index);
        return current_node;
    }

    if(cmp(minT.x, mx) >= 0) child_index += 1;
    if(cmp(minT.y, my) >= 0) child_index += 2;
    if(cmp(minT.z, mz) >= 0) child_index += 4;

    int next_node = nodes[current_node].getChild(child_index);

    if(next_node == -1) {
        next_node = nodes.size();
        OctreeNode tmp_node = createNode(current_node, child_index);
        nodes.push_back(tmp_node);
        nodes[current_node].setChild(child_index, next_node);
    }

    return add(triangle, t_index, next_node);
}

void Octree::find(const Ray &ray, double &current_t, int current_node, std::vector<int> &candidates) const {
    const std::vector<int> tmp = nodes[current_node].getSurfaces();
    for(int i=0;i<tmp.size();i++) candidates.push_back(tmp[i]);

    double mx = nodes[current_node].getMX();
    double my = nodes[current_node].getMY();
    double mz = nodes[current_node].getMZ();

    Point current_point = ray.pointAt(current_t);

    while(nodes[current_node].isInside(current_point)) {
        int child_index = 0;

        if(cmp(current_point.x, mx) == 1) child_index += 1;
        if(cmp(current_point.y, my) == 1) child_index += 2;
        if(cmp(current_point.z, mz) == 1) child_index += 4;

        int next_node = nodes[current_node].getChild(child_index);
        if(next_node != -1) find(ray, current_t, next_node, candidates);

        double delta = nodes[current_node].moveToNext(ray, current_t);
        current_t += delta;
        current_point = ray.pointAt(current_t);
    }
}

int Octree::add(const Triangle &triangle, int t_index) {
    return add(triangle, t_index, 0);
}

std::vector<int> Octree::find(const Ray &ray) const {
    std::vector<int> candidates;
    double t = 0.0;
    find(ray, t, 0, candidates);
    return candidates;
}
