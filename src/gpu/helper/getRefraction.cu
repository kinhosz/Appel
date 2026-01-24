#include <Appel/gpu/helper.h>

namespace Appel {
    __device__ GPoint getRefraction(GTriangle surface, GPoint dir, 
        float refIndex) {

        GPoint tNormal = normalize(getNormal(surface));
        GPoint tDir = normalize(dir);

        float PI = acos(-1.0);
        float theta = angle(tNormal, tDir);

        if(f_cmp(theta, PI/2.0) == 1) tNormal = multByScalar(tNormal, -1.0);

        float theta1 = angle(tDir, tNormal);
        float cosTheta1 = cos(theta1);

        float ref = 1.0/refIndex;

        float sinTheta1_2 = 1.0 - cosTheta1 * cosTheta1;
        float cosTheta2 = sqrt(1.0 - ref*ref * sinTheta1_2);

        GPoint tmp1 = multByScalar(tDir, -1.0 * ref);
        GPoint tmp2 = multByScalar(tNormal, ref * cosTheta1 - cosTheta2);

        tmp1 = add(tmp1, tmp2);

        return normalize(tmp1);
    }
}
