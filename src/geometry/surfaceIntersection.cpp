#include <geometry/surfaceIntersection.h>
#include <geometry/utils.h>
#include <math.h>

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

Vetor SurfaceIntersection::getReflection(const Vetor &direction) const {
    Vetor tNormal = normal.normalize();
    Vetor tDirection = normal.normalize();

    return (Vetor(tNormal * (2.0 * tNormal.dot(tDirection))) - tDirection).normalize();   
}

Vetor SurfaceIntersection::getRefraction(Vetor direction, double refractionIndex) const {
    Vetor tNormal = normal.normalize();
    Vetor tDirection = direction.normalize();

    if(cmp(tNormal.angle(tDirection), PI/2.0) == 1) tNormal = tNormal * -1.0;

    double theta1 = tDirection.angle(tNormal);
    double cosTheta1 = cos(theta1);

    double ref = 1.0/refractionIndex;

    double sinTheta1_2 = 1.0 - cosTheta1 * cosTheta1;

    double cosTheta2 = sqrt(1.0 - ref*ref * sinTheta1_2);

    Vetor refraction = ((tDirection * -1.0) * ref) + (tNormal * (ref * cosTheta1 - cosTheta2));

    return refraction.normalize();
}
