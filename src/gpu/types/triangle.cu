#include <gpu/types/triangle.h>
#include <geometry/triangle.h>

GTriangle::GTriangle(const Triangle &t, int host_id) {
    for(int i=0;i<3;i++) {
        this->point[i] = GPoint(t.getVertex(i));
    }
    this->host_id = host_id;
}
