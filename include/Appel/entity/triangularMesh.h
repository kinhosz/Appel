#ifndef ENTITY_TRIANGULARMESH_H
#define ENTITY_TRIANGULARMESH_H

#include <vector>
#include <string>
#include <Appel/entity/box.h>
#include <Appel/geometry/triangle.h>
#include <Appel/graphic/image.h>

namespace Appel {
    class TriangularMesh : public Box {
    private:
        std::vector<Triangle> triangles;
        bool _hasTexture;
        Image texture;
        Point position;
    public:
        TriangularMesh();
        TriangularMesh(const std::vector<Triangle> triangles);
        TriangularMesh(std::string filename);

        bool hasTexture() const;
        void setTexture(std::string filename);
        Pixel getTexture(double x, double y) const;
        Triangle getTriangle(int index) const;
        std::vector<Triangle> getTriangles() const;
        SurfaceIntersection intersect(const Ray& ray) const override;
        virtual Point getPosition() const override;
        virtual void moveTo(const Point &p) override;
    };
}

#endif
