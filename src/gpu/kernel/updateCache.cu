#include <Appel/gpu/kernel.h>

__global__ void updateCache(int device_id, GTriangle triangle, GTriangle* cache) {
    cache[device_id] = triangle;
}
