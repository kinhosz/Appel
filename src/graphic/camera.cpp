#include <graphic/camera.h>

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

Frame Camera::take(Scene &scene) const {
    Frame frame(vPixels, hPixels);

    for(int x=0; x<hPixels; x++) {
        for(int y=0; y<vPixels; y++) {
            Ray ray = createRay(x, y);
            Color color = scene.traceRay(ray, 0);

            frame.setPixel(x, (vPixels - y - 1), Pixel(color));
        }
    }

    return frame;
}

Frame Camera::resize(const Frame &frame, double r) const {
    int vRes = (double)frame.vertical() * r;
    int hRes = (double)frame.horizontal() * r;

    Frame frameResized(vRes, hRes);
    double pixelsPerV = (double)vRes / (frame.vertical() - 1);
    double pixelsPerH = (double)hRes / (frame.horizontal() - 1);

    for(int x=0; x<hRes; x++) {
        for(int y=0; y<vRes; y++) {
            int fx = x / pixelsPerH;
            int fy = y / pixelsPerV;

            double factor_x = (double)(x - fx * pixelsPerH) / pixelsPerH;
            double factor_y = (double)(y - fy * pixelsPerV) / pixelsPerV;

            Color color;
            color.setRed(
                (1.0 - factor_x) * (
                    (1.0 - factor_y) * (double)frame.getPixel(fx, fy).getRed()/255
                    + (factor_y) * (double)frame.getPixel(fx, fy+1).getRed()/255
                )
                + (factor_x) * (
                    (1.0 - factor_y) * (double)frame.getPixel(fx+1, fy).getRed()/255
                    + (factor_y) * (double)frame.getPixel(fx+1, fy+1).getRed()/255
                )
            );
            color.setGreen(
                (1.0 - factor_x) * (
                    (1.0 - factor_y) * (double)frame.getPixel(fx, fy).getGreen()/255
                    + (factor_y) * (double)frame.getPixel(fx, fy+1).getGreen()/255
                )
                + (factor_x) * (
                    (1.0 - factor_y) * (double)frame.getPixel(fx+1, fy).getGreen()/255
                    + (factor_y) * (double)frame.getPixel(fx+1, fy+1).getGreen()/255
                )
            );
            color.setBlue(
                (1.0 - factor_x) * (
                    (1.0 - factor_y) * (double)frame.getPixel(fx, fy).getBlue()/255
                    + (factor_y) * (double)frame.getPixel(fx, fy+1).getBlue()/255
                )
                + (factor_x) * (
                    (1.0 - factor_y) * (double)frame.getPixel(fx+1, fy).getBlue()/255
                    + (factor_y) * (double)frame.getPixel(fx+1, fy+1).getBlue()/255
                )
            );

            frameResized.setPixel(x, y, Pixel(color));
        }
    }

    return frameResized;
}
