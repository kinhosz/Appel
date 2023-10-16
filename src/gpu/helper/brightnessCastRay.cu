#include <gpu/helper.h>

__device__ void brightnessCastRay(int pos, GTriangle surface, int lightId,
    GRay ray, GTriangle* cache_triangle, GPoint* cache_light, int* buffer) {
    
    float PI = acos(-1.0);

    GPoint normal = getNormal(surface);

    float theta = angle(ray.direction, multByScalar(normal, -1.0));
    if(f_cmp(theta, PI/2.0) == 1) multByScalar(normal, -1.0);

    GPoint match = pointAt(ray, getDistance(ray, surface));

    GPoint dir = normalize(sub(cache_light[lightId], match));

    GRay tmp;
    tmp.direction = dir;
    tmp.location = match;

    tmp.location = pointAt(tmp, 0.01);

    int device_id = castRay(tmp, cache_triangle);
    
    if(device_id == -1) {
        buffer[pos] = -1;
    }
    else {
        buffer[pos] = cache_triangle[pos].host_id;
    }
}
