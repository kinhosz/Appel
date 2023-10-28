#include <gpu/helper.h>

__device__ int isInside(GPoint point, GTriangle triangle) {
    float ABC = 0.5 * norm(cross(sub(triangle.point[1],triangle.point[0]), sub(triangle.point[2], triangle.point[0])));
    float PBC = 0.5 * norm(cross(sub(triangle.point[1], point), sub(triangle.point[2], point)));
    float PAC = 0.5 * norm(cross(sub(triangle.point[0], point), sub(triangle.point[2], point)));
    float PAB = 0.5 * norm(cross(sub(triangle.point[0], point), sub(triangle.point[1], point)));

    float u = PBC / ABC;
    float v = PAC / ABC;
    float w = PAB / ABC;

    float sumt = u + v + w;

    int isnotinside = 0;
    isnotinside += (f_cmp(sumt, 1.0) != 0 ? 1-isnotinside: 0);
    isnotinside += (f_cmp(u, 0.0) == -1 ? 1-isnotinside: 0);
    isnotinside += (f_cmp(u, 1.0) == 1 ? 1-isnotinside: 0);
    isnotinside += (f_cmp(v, 0.0) == -1 ? 1-isnotinside: 0);
    isnotinside += (f_cmp(v, 1.0) == 1 ? 1-isnotinside: 0);
    isnotinside += (f_cmp(w, 0.0) == -1 ? 1-isnotinside: 0);
    isnotinside += (f_cmp(w, 1.0) == 1 ? 1-isnotinside: 0);

    return 1-isnotinside;
}
