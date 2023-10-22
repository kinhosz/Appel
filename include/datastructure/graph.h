#ifndef GRAPH_DATASTRUCTURE_H
#define GRAPH_DATASTRUCTURE_H

#include <vector>

class Graph {
    std::vector<std::vector<int>> edge;
    std::vector<int> value;

public:
    Graph();
    int addEdge(int parent, int val);
    void dfs(int node, std::vector<int> &path);
};

#endif
