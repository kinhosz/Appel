#ifndef GEOMETRY_TRIANGLE_H
#define GEOMETRY_TRIANGLE_H

#include <cstddef>
#include <geometry/point.h>
#include <geometry/vetor.h>
#include <assert.h> 

struct Triangle {
    Point vertices[3]; 
    Color color;

    Triangle();
    Triangle(const Point& v1, const Point& v2, const Point& v3, const Color& color);

    double area() const;
    Point centroid() const;
    Vetor normal() const;
    bool contains(const Point& p) const; 

    const Point& getVertex(int index) const {
        assert(index >= 0 && index < 3);
        return vertices[index];
    }

    bool operator==(const Triangle& other) const;
    bool operator!=(const Triangle& other) const;
};

#endif
