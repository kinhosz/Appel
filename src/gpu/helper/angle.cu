#include <Appel/gpu/helper.h>

__device__ float angle(GPoint a, GPoint b) {
    float eps = 1e-9;

    float cost = dot(a, b) / (norm(a) * norm(b));
    cost = min(max(cost, -1.0 + eps), 1.0 - eps);

    return acos(cost);
}
