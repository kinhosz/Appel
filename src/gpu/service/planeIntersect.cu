#include <gpu/service.h>

__device__ void vectorCross(
    float ax, float ay, float az, float bx, float by, float bz,
    float *x, float *y, float *z) {

    (*x) = (ay * bz) - (az * by);
    (*y) = (az * bx) - (ax * bz);
    (*z) = (ax * by) - (ay * bx);
}

__device__ void buildVector(GTriangle gt, float *vx, float *vy, float *vz) {
    float t0x = (gt.p1x - gt.p0x), t0y = (gt.p1y - gt.p0y), t0z = (gt.p1z - gt.p0z);
    float t1x = (gt.p2x - gt.p0x), t1y = (gt.p2y - gt.p0y), t1z = (gt.p2z - gt.p0z);

    vectorCross(t0x, t0y, t0z, t1x, t1y, t1z, vx, vy, vz);
}

__device__ bool isOrthogonal(float ax, float ay, float az, float bx, float by, float bz) {
    float scalar = ax*bx + ay*by + az*bz;

    return (scalar < 1e-9);
}

__device__ float planeIntersect(GRay ray, GTriangle gt) {
    float vx, vy, vz;
    buildVector(gt, &vx, &vy, &vz);

    if(isOrthogonal(vx, vy, vz, ray.dx, ray.dy, ray.dz)) return;

    double D = -vx * gt.p0x - vy * gt.p0y - vz * gt.p0z;
    double A = vx, B = vy, C = vz;

    double c1 = (A * ray.lx + B * ray.ly + C * ray.lz + D);
    double c2 = (A * ray.dx + B * ray.dy + C * ray.dz);

    double t = -c1/c2;

    if(t < 1e-9) return -1.0;

    return t;
}
