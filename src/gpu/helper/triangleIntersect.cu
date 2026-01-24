#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ float triangleIntersect(GRay ray, GTriangle triangle) {
        if(triangle.host_id == -1) return -1.0;

        GPoint normal = getNormal(triangle);
        if(f_cmp(dot(normal, ray.direction), 0.0) == 0) return -1.0;

        float curDist = getDistance(ray, triangle);
        if(f_cmp(curDist, 0) <= 0) return -1.0;

        if(!isInside(pointAt(ray, curDist), triangle)) return -1.0;

        return curDist;
    }
}
