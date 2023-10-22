#include <datastructure/graph.h>
#include <stdexcept>

Graph::Graph() {
    edge.push_back(std::vector<int>());
    value.push_back(-1);
}

int Graph::addEdge(int parent, int val) {
    if(parent < 0 || parent >= (int)edge.size()) {
        throw std::runtime_error("Parent greater than edges");
    }

    int newVertex = (int)edge.size();
    edge.push_back(std::vector<int>());
    value.push_back(val);

    edge[parent].push_back(newVertex);

    return newVertex;
}

void Graph::dfs(int node, std::vector<int> &path) {
    if(node != 0) path.push_back(value[node]);

    for(int child: edge[node]) dfs(child, path);
}