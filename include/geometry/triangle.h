#ifndef GEOMETRY_TRIANGLE_H
#define GEOMETRY_TRIANGLE_H

#include <geometry/point.h>
#include <geometry/vetor.h>
#include <graphic/color.h>

struct Triangle {
    Point vertices[3]; 
    Color color;

    Triangle();
    Triangle(Point v1, Point v2, Point v3, Color color);

    double area() const;
    Point centroid() const;
    Vetor normal() const;

    Point getVertex(int index) const {
        assert(index >= 0 && index < 3);
        return vertices[index];
    }

    bool operator==(const Triangle& other) const;
    bool operator!=(const Triangle& other) const;
};

#endif
