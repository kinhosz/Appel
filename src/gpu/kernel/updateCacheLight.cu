#include <gpu/kernel.h>

__global__ void updateCacheLight(int device_id, GPoint p, GPoint* light) {
    light[device_id] = p;
}
