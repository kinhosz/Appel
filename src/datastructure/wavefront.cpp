#include <Appel/datastructure/wavefront.h>
#include <fstream>
#include <assert.h>

Wavefront::Wavefront() {}

bool Wavefront::isComment(const std::string &line) const {
  return line.size() > 0 && line[0] == '#';
}

bool Wavefront::isVertice(const std::string &line) const {
  return line.size() > 1 && line[0] == 'v' && line[1] == ' ';
}

bool Wavefront::isTexture(const std::string &line) const {
  return line.size() > 1 && line[0] == 'v' && line[1] == 't';
}

bool Wavefront::isNormal(const std::string &line) const {
  return line.size() > 1 && line[0] == 'v' && line[1] == 'n';
}

bool Wavefront::isParameter(const std::string &line) const {
  return line.size() > 1 && line[0] == 'v' && line[1] == 'p';
}

bool Wavefront::isPolygonal(const std::string &line) const {
  return line.size() > 0 && line[0] == 'f';
}

std::vector<std::string> Wavefront::getArgs(const std::string &line) const {
  std::string tmp = "";
  std::vector<std::string> args;

  for(int i=0;i<(int)line.size();i++) {
    if(line[i] == ' ') {
      if(tmp.size() > 0) args.push_back(tmp);
      tmp = "";
    } else tmp += line[i];
  }

  if(tmp.size() > 0) args.push_back(tmp);

  return args;
}

std::vector<int> Wavefront::getIndexes(const std::string &arg) const {
  std::string tmp = "";
  std::vector<int> indexes;

  for(int i=0;i<(int)arg.size();i++) {
    if(arg[i] == '/') {
      if(tmp == "") indexes.push_back(0);
      else indexes.push_back(std::stoi(tmp));
      tmp = "";
    } else tmp += arg[i];
  }

  if(tmp == "") indexes.push_back(0);
  else indexes.push_back(std::stoi(tmp));

  return indexes;
}

Point Wavefront::getVertice(const std::string &line) const {
  const std::vector<std::string> &args = getArgs(line);

  double p0 = std::stod(args[1]);
  double p1 = std::stod(args[2]);
  double p2 = std::stod(args[3]);

  return Point(p0, p1, p2);
}

std::pair<double, double> Wavefront::getVerticeTexture(const std::string &line) const {
  const std::vector<std::string> &args = getArgs(line);

  double u = std::stod(args[1]);
  double v = std::stod(args[2]);

  return std::make_pair(u, v);
}

std::vector<Triangle> Wavefront::getPolygon(const std::string &line) const {
  const std::vector<std::string> &args = getArgs(line);
  std::vector<Triangle> polygon;

  std::vector<Point> points;
  std::vector<std::pair<double, double>> vts;

  for(int i=1;i<(int)args.size();i++) {
    std::vector<int> idxs = getIndexes(args[i]);

    int p_idx;
    if(idxs[0] < 0) p_idx = (int)vertices.size() + idxs[0];
    else p_idx = idxs[0] - 1;

    int vt_idx;
    if(idxs[1] < 0) vt_idx = (int)textures.size() + idxs[1];
    else vt_idx = idxs[1] - 1;

    points.push_back(vertices[p_idx]);
    vts.push_back((vt_idx == -1 ? std::pair(0.0, 0.0) : textures[vt_idx]));
  }

  for(int i=2;i<(int)points.size();i++) {
    Triangle t(points[0], points[i-1], points[i], Color(255, 0, 0));
    t.setUVMapping(vts[0], vts[i-1], vts[i]);
    polygon.push_back(t);
  }

  return polygon;
}

Wavefront::Wavefront(const std::string &filename) {
  std::ifstream f_obj(filename);
  assert(f_obj.is_open());

  std::string line;
  while (std::getline(f_obj, line)) {
    if(isVertice(line)) {
      vertices.push_back(getVertice(line));

    } else if(isTexture(line)) {
      textures.push_back(getVerticeTexture(line));

    } else if(isPolygonal(line)) {
      const std::vector<Triangle> polygon = getPolygon(line);
      for(int i=0;i<(int)polygon.size();i++) {
        triangles.push_back(polygon[i]);
      }
    }
  }
}

std::vector<Point> Wavefront::getVertices() const {
  return vertices;
}

std::vector<Triangle> Wavefront::getTriangles() const {
  return triangles;
}
