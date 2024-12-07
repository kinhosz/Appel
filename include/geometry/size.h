#ifndef GEOMETRY_SIZE_H
#define GEOMETRY_SIZE_H

struct Size {
    int width;
    int height;

    Size();
    Size(int width, int height);

    void setWidth(int width);
    void setHeight(int height);

    bool operator==(const Size& other) const;
    bool operator!=(const Size& other) const;
};

#endif
