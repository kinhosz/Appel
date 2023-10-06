#include <gpu/manager.h>
#include <gpu/types/ray.h>

Manager::Manager() {
    calls_counter = 0;
    cache_limit = 100000;
    tableFrequency = std::set<std::pair<int, int>>();
    hostToDeviceID = std::map<int, int>();

    cudaMalloc(&cache, cache_limit);

    for(int i=0;i<cache_limit;i++) {
        GTriangle gt;
        gt.host_id = -1;
        cudaMemcpy(&cache[i], &gt, 1, cudaMemcpyHostToDevice);
    }
}

void Manager::transfer(int host_id, const Triangle &triangle) {
    if(isOnCache(host_id)) return;

    int device_id = getFreeDeviceId();
    GTriangle gt;
    gt.p0x = triangle.getVertex(0).x;
    gt.p0y = triangle.getVertex(0).y;
    gt.p0z = triangle.getVertex(0).z;

    gt.p1x = triangle.getVertex(1).x;
    gt.p1y = triangle.getVertex(1).y;
    gt.p1z = triangle.getVertex(1).z;

    gt.p2x = triangle.getVertex(2).x;
    gt.p2y = triangle.getVertex(2).y;
    gt.p2z = triangle.getVertex(2).z;

    gt.host_id = host_id;

    lazy.push_back(std::make_pair(device_id, gt));
}

bool Manager::isOnCache(int host_id) {
    auto it = tableFrequency.lower_bound(std::make_pair(host_id, -1));
    if(it == tableFrequency.end() || (*it).first != host_id) return false;

    tableFrequency.erase(it);
    tableFrequency.insert(std::make_pair(host_id, calls_counter));

    hostToDeviceID[host_id] = calls_counter;

    return true;
}

int Manager::getFreeDeviceId() {
    if(tableFrequency.size() < cache_limit) return tableFrequency.size();

    auto it = tableFrequency.begin();
    int host_id = (*it).first;
    int device_id = hostToDeviceID.at(host_id);
    
    hostToDeviceID.erase(host_id);
    tableFrequency.erase(it);

    return device_id;
}

void Manager::pendingTransfer() {
    cudaStream_t streams[lazy.size()];

    for(int i=0;i<lazy.size();i++) {
        int dvc_id = lazy[i].first;
        GTriangle gt = lazy[i].second;

        cudaMemcpyAsync(&cache[dvc_id], &gt, 1, cudaMemcpyHostToDevice, streams[i]);
    }

    for(int i=0;i<lazy.size();i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    lazy.clear();
}

int Manager::run(const Ray &ray) {
    pendingTransfer();
    GRay gr;
    gr.lx = ray.location.x;
    gr.ly = ray.location.y;
    gr.lz = ray.location.z;

    gr.dx = ray.direction.x;
    gr.dy = ray.direction.y;
    gr.dz = ray.direction.z;

    GRay *dvc_gr;
    cudaMalloc(&dvc_gr, 1);
    cudaMemcpy(&dvc_gr, &gr, 1, cudaMemcpyHostToDevice);

    int idx = -1;
    float dist = 0.0;

    int *dvc_idx;
    float *dvc_dist;

    cudaMalloc(&dvc_idx, 1);
    cudaMemcpy(&dvc_idx, &idx, 1, cudaMemcpyHostToDevice);

    cudaMalloc(&dvc_dist, 1);
    cudaMemcpy(&dvc_dist, &dist, 1, cudaMemcpyHostToDevice);

    // call here the kernel
}