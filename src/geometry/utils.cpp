#include <geometry/utils.h>
#include <cmath>

const double EPSILON = 1e-12;

int cmp(double a, double b){
    if(abs(a - b) < EPSILON) return 0;
    return (a < b ? -1 : 1);
}