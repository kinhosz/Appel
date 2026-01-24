#include <Appel/gpu/helper.h>

__device__ GPoint getNormal(GTriangle triangle) {
    GPoint a = sub(triangle.point[1], triangle.point[0]);
    GPoint b = sub(triangle.point[2], triangle.point[0]);

    GPoint c = normalize(cross(a, b));

    return c;
}
