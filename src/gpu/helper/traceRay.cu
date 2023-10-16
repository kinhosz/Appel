#include <gpu/helper.h>

__device__ void traceRay(const int offset, int node, int level, GRay ray,
    GTriangle* cache_triangle, GPoint* cache_light, const int depth, 
    const int lights, int* buffer) {

    if(level >= depth) return;

    int device_id = castRay(ray, cache_triangle);

    int current_offset = offset + node * (lights + 1);

    if(device_id == -1) {
        buffer[current_offset] = -1;
        return;
    }

    const GTriangle surface = cache_triangle[device_id];

    buffer[current_offset] = surface.host_id;

    for(int i=1;i<=lights;i++) {
        brightnessCastRay(current_offset + i, surface, i,
            ray, cache_triangle, cache_light, buffer);
    }

    int reflectionNode = (node * 2) + 1;
    int refractionNode = (node * 2) + 2;

    float distance = getDistance(ray, surface);

    GRay reflectionRay;
    reflectionRay.location = pointAt(ray, distance - 0.01);
    reflectionRay.direction = getReflection(surface, multByScalar(ray.direction, -1.0));

    traceRay(offset, reflectionNode, level+1, reflectionRay, cache_triangle,
        cache_light, depth, lights, buffer);
    
    GRay refractionRay;
    refractionRay.location = pointAt(ray, distance + 0.01);
    // TODO: change refraction index 
    refractionRay.direction = getRefraction(surface, multByScalar(ray.direction, -1.0), 1.0);

    traceRay(offset, refractionNode, level+1, refractionRay, cache_triangle,
        cache_light, depth, lights, buffer);
}
