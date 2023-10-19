#include <gpu/helper.h>

__device__ GPoint getReflection(GTriangle triangle, GPoint dir) {
    GPoint tNormal = normalize(getNormal(triangle));
    GPoint tDir = normalize(dir);

    float m = 2.0 * dot(tNormal, tDir);
    GPoint ref = sub(multByScalar(tNormal, m), tDir);

    return normalize(ref);
}
