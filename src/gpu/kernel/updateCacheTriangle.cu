#include <gpu/kernel.h>

__global__ void updateCacheTriangle(int device_id, GTriangle triangle, GTriangle* cache) {
    cache[device_id] = triangle;
}
