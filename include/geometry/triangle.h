#ifndef GEOMETRY_TRIANGLE_H
#define GEOMETRY_TRIANGLE_H

#include <geometry/point.h>
#include <geometry/vetor.h>

struct Triangle {
    Point vertices[3]; 

    Triangle();
    Triangle(const Point& v1, const Point& v2, const Point& v3);

    double area() const;
    Point centroid() const;
    Vetor normal() const;
    bool contains(const Point& p) const; 

    bool operator==(const Triangle& other) const;
    bool operator!=(const Triangle& other) const;
};

#endif
