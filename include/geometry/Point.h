#ifndef GEOMETRY_POINT_H
#define GEOEMTRY_POINT_H

struct Point {
    double x, y, z;
    
    Point();
    Point(double x, double y, double z);

    double distance(const Point &other);
};

#endif