#include <Appel/gpu/types/triangle.h>
#include <Appel/geometry/triangle.h>

__host__ __device__ GTriangle::GTriangle() {
    this->host_id = -1;
}

GTriangle::GTriangle(const Triangle &t, int host_id) {
    for(int i=0;i<3;i++) {
        this->point[i] = GPoint(t.getVertex(i));
    }
    this->host_id = host_id;
}
