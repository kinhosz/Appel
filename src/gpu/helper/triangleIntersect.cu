#include <gpu/helper.h>

__device__ float triangleIntersect(GRay ray, GTriangle triangle) {
    GPoint normal = getNormal(triangle);
    float curDist = getDistance(ray, triangle, normal);
    float ang = dot(normal, ray.direction);
    int isd = isInside(pointAt(ray, curDist), triangle);

    if(triangle.host_id == -1 ||
        f_cmp(ang, 0.0) == 0 ||
        f_cmp(curDist, 0) <= 0 ||
        isd == 0) {

        return __FLT_MAX__;
    }

    return curDist;
}
