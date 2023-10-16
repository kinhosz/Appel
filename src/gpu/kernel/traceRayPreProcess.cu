#include <gpu/kernel.h>
#include <gpu/helper.h>

__global__ void traceRayPreProcess(const GPoint up, const GPoint right, const GPoint front,
    const GPoint loc, const float dist, const int height, const int width,
    GTriangle* cache_triangle, GPoint* cache_light, const int depth, const int lights,
    int* buffer) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread_id >= height * width) return;

    GRay ray = createRay(up, right, front, loc, dist, height, width, thread_id);

    int offset = (1<<depth) * (lights + 1) * thread_id;

    traceRay(offset, 0, 0, ray, cache_triangle, cache_light, depth, lights, buffer);
}
