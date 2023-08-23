#ifndef GEOMETRY_TRIANGLE_H
#define GEOMETRY_TRIANGLE_H

#include <geometry/point.h>
#include <geometry/vetor.h>
#include <geometry/surfaceIntersection.h>
#include <geometry/ray.h>
#include <graphic/color.h>
#include <assert.h> 

struct Triangle {
    Point vertices[3];
    Vetor vertexNormals[3];
    Vetor triangleNormal;
    Color color;

    Triangle();
    Triangle(Point v1, Point v2, Point v3, Vetor n1, Vetor n2, Vetor n3, Vetor triangleNormal, Color color);

    double area() const;
    Point centroid() const;

    Point getVertex(int index) const {
        assert(index >= 0 && index < 3);
        return vertices[index];
    }

    Vetor getVertexNormal(int index) const {
        assert(index >= 0 && index < 3);
        return vertexNormals[index];
    }

    Vetor getTriangleNormal() const {
        return triangleNormal;
    }

    bool operator==(const Triangle& other) const;
    bool operator!=(const Triangle& other) const;

    bool isInside(const Point &p) const;

    SurfaceIntersection intersect(Ray ray) const;
};

#endif
