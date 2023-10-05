#include <graphic/camera.h>
#include <iostream>

Camera::Camera(Point loc, Point focus, int vPixels, int hPixels): 
    Camera(loc, focus, vPixels, hPixels, (double) (hPixels - 1) / 2.0) {}

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

void Camera::move(Point p) {
    location.x += p.x;
    location.y += p.y;
    location.z += p.z;
}

Ray Camera::createRay(int x, int y) const {
    double hPivot = hPixels/2;
    double vPivot = vPixels/2;

    Vetor pixelUp = vUp * (y - vPivot);
    Vetor pixelRight = vRight * (x - hPivot);

    Vetor pixelFront = vFront * distance;

    Vetor pixelVector = (pixelFront + pixelRight + pixelUp).normalize();

    return Ray(location, pixelVector);
}

Frame Camera::take(const Scene &scene) const {
    Frame frame(vPixels, hPixels);

    for(int x=0; x<hPixels; x++) {
        for(int y=0; y<vPixels; y++) {
            Ray ray = createRay(x, y);
            Color color = scene.traceRay(ray, 0);

            frame.setPixel(x, (vPixels - y - 1), Pixel(color));

            double perc = (double)(x * vPixels + y) / (hPixels * vPixels);
            perc *= 100;

            if((x * hPixels + y)% 1000 == 0) {
                std::cerr << "perc = " << perc << "\n";
            }
        }
    }

    return frame;
}
