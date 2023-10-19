#include <gpu/reducer.h>

__global__ void getMin(float* buffer_dist, int* buffer_idx, int* N, int* res) {
    float val = buffer_dist[0];
    int idx = buffer_idx[0];

    for(int i=1;i<(*N);i++) {
        if(buffer_idx[i] == -1) continue;
        if(idx == -1 || val > buffer_dist[i]) {
            idx = buffer_idx[i];
            val = buffer_dist[i];
        }
    }

    *res = idx;
}
