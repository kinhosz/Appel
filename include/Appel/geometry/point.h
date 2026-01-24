#ifndef GEOMETRY_POINT_H
#define GEOMETRY_POINT_H

namespace Appel {
    struct Point {
        double x, y, z;
        
        Point();
        Point(double x, double y, double z);

        double distance(const Point &other) const;

        bool operator>(const Point &other) const;
        bool operator<(const Point &other) const;
        bool operator==(const Point &other) const;
        bool operator!=(const Point &other) const;
    };
}

#endif
