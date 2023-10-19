#include <gpu/helper.h>

__device__ int castRay(GRay ray, GTriangle* cache_triangle) {
    float minDist = -1.0;
    int idx = -1;

    int i = 0;
    while(cache_triangle[i].host_id != -1) {
        GPoint normal = getNormal(cache_triangle[i]);
        if(f_cmp(dot(normal, ray.direction), 0.0) == 0) {
            i++;
            continue;
        }

        float curDist = getDistance(ray, cache_triangle[i]);
        if(f_cmp(curDist, 0) <= 0 || 
            (f_cmp(curDist, minDist) >= 0 && idx != -1)) {
            i++;
            continue;
        }

        if(!isInside(pointAt(ray, curDist), cache_triangle[i])) {
            i++;
            continue;
        }

        if(idx == -1 || f_cmp(curDist, minDist) == -1) {
            minDist = curDist;
            idx = i;
        }
        i++;
    }

    return idx;
}
