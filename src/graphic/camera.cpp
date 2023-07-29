#include <graphic/camera.h>

Camera::Camera(Point loc, Point focus, int vPixels, int hPixels) {
    double dist = (double) (hPixels - 1) / 2.0;

    Camera(loc, focus, vPixels, hPixels, dist);
}

Camera::Camera(Point loc, Point focus, int vPixels, int hPixels, double dist) {
    this->location = loc;
    this->focus = focus;
    this->vPixels = vPixels;
    this->hPixels = hPixels;
    this->distance = dist;

    Vetor vFocus(focus);
    Vetor vLoc(loc);

    vUp = Vetor(0, 0, 1);
    vFront = vFocus - vLoc;

    vRight = vFront.cross(vUp);
    vUp = vRight.cross(vFront);

    this->vUp = vUp.normalize();
    this->vFront = vFront.normalize();
    this->vRight = vRight.normalize();
}

void Camera::zoom(double delta) {
    distance += delta;
}

void Camera::move(Point &p) {
    location.x += p.x;
    location.y += p.y;
    location.z += p.z;
}