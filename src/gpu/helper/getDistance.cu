#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ float getDistance(GRay ray, GTriangle triangle) {
        GPoint normal = getNormal(triangle);

        float D = -normal.x * triangle.point[0].x - normal.y * triangle.point[0].y - normal.z * triangle.point[0].z;
        float A = normal.x, B = normal.y, C = normal.z;

        float c1 = (A * ray.location.x + B * ray.location.y + C * ray.location.z + D);
        float c2 = (A * ray.direction.x + B * ray.direction.y + C * ray.direction.z);

        float t = -c1/c2;

        return t;
    }
}
