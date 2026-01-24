#ifndef WAVEFRONT_DATASTRUCTURE_H
#define WAVEFRONT_DATASTRUCTURE_H

#include <STB/stb_image.h>

#include <string>
#include <vector>
#include <Appel/geometry/point.h>
#include <Appel/geometry/triangle.h>

namespace Appel {
  class Wavefront {
  private:
    std::vector<Point> vertices;
    std::vector<std::pair<double, double>> textures;
    std::vector<Triangle> triangles;

    std::vector<std::string> getArgs(const std::string &line) const;
    std::vector<int> getIndexes(const std::string &arg) const;

    bool isComment(const std::string &line) const;
    bool isVertice(const std::string &line) const;
    bool isTexture(const std::string &line) const;
    bool isNormal(const std::string &line) const;
    bool isParameter(const std::string &line) const;
    bool isPolygonal(const std::string &line) const;

    Point getVertice(const std::string &line) const;
    std::pair<double, double> getVerticeTexture(const std::string &line) const;
    std::vector<Triangle> getPolygon(const std::string &line) const;

  public:
    Wavefront();
    Wavefront(const std::string &filename);

    std::vector<Point> getVertices() const;
    std::vector<Triangle> getTriangles() const;
  };
}

#endif
