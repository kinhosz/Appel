#ifndef GRAPHIC_CAMERA_H
#define GRAPHIC_CAMERA_H

#include <geometry/point.h>
#include <geometry/vetor.h>
#include <graphic/frame.h>
#include <entity/scene.h>

class Camera {
private:
    Point location;
    Point focus;
    Vetor vUp, vFront, vRight;
    double distance;
    int vPixels;
    int hPixels;

    Ray createRay(int x, int y) const;

public:
    Camera(Point loc, Point focus, int vPixels, int hPixels);
    Camera(Point loc, Point focus, int vPixels, int hPixels, double dist);

    void setResolution(int hPixels, int vPixels);

    void zoom(double delta);
    void move(Point p);
    void setPosition(Point p);

    Frame take(Scene &scene) const;
    Frame resize(const Frame &frame, int hRes, int vRes) const;
};

#endif
