#ifndef GEOMETRY_TRIANGLE_H
#define GEOMETRY_TRIANGLE_H

#include <Appel/geometry/point.h>
#include <Appel/geometry/vetor.h>
#include <Appel/geometry/surfaceIntersection.h>
#include <Appel/geometry/ray.h>
#include <Appel/geometry/coordinateSystem.h>
#include <Appel/graphic/color.h>
#include <assert.h> 
#include <utility>

namespace Appel {
    struct Triangle {
        Point vertices[3];
        Vetor vertexNormals[3];
        Vetor triangleNormal;
        std::pair<double, double> uv[3];
        Color color;

        Triangle();
        Triangle(Point v1, Point v2, Point v3, Color triangleColor);
        Triangle(Point v1, Point v2, Point v3, Vetor n1, Vetor n2, Vetor n3, Vetor triangleNormal, Color color);

        void setUVMapping(std::pair<double, double> vt1, std::pair<double, double> vt2, std::pair<double, double> vt3);

        double area() const;
        Point centroid() const;

        Point getVertex(int index) const;
        Vetor getVertexNormal(int index) const;
        Vetor getTriangleNormal() const;

        Point getMin() const;
        Point getMax() const;

        bool operator==(const Triangle& other) const;
        bool operator!=(const Triangle& other) const;

        bool isInside(const Point &p) const;
        std::pair<double, double> getUVAtPoint(const Point &p) const;

        SurfaceIntersection intersect(Ray ray) const;
        SurfaceIntersection getSurface(Ray ray) const;

        Triangle rebase(const CoordinateSystem& cs) const;
        Triangle moveTo(const Point &p) const;
        Triangle rotate(double alphaX, double alphaY, double alphaZ) const;
    };
}

#endif
