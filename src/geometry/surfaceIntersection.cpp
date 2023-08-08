#include <geometry/surfaceIntersection.h>
#include <geometry/utils.h>

SurfaceIntersection::SurfaceIntersection(){
    color = Color(0, 0, 0);
    distance = DOUBLE_INF;
    normal = Vetor(0, 0, 1);
}

SurfaceIntersection::SurfaceIntersection(
    Color color, double distance, Vetor normal){
    
    this->color = color;
    this->distance = distance;
    this->normal = normal.normalize();
}
