#include <gpu/helper.h>

__device__ GRay createRay(const GPoint up, const GPoint right, 
    const GPoint front, const GPoint loc, const float dist,
    const int height, const int width, int i) {

    float vPivot = (float)height/2.0;
    float hPivot = (float)width/2.0;

    int x = i/height;
    int y = i%height;

    GPoint pixel_up = multByScalar(up, (float)y - vPivot);
    GPoint pixel_right = multByScalar(right, (float)x - hPivot);
    GPoint pixel_front = multByScalar(front, dist);

    pixel_up.x += (pixel_right.x + pixel_front.x);
    pixel_up.y += (pixel_right.y + pixel_front.y);
    pixel_up.z += (pixel_right.z + pixel_front.z);

    pixel_up = normalize(pixel_up);

    GRay ray;
    ray.direction = pixel_up;
    ray.location = loc;

    return ray;
}