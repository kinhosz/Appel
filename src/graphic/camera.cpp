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


#ifdef APPEL_GPU_DISABLED

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

#else

Frame Camera::take(Scene& scene) const {
    int BATCH_SIZE = scene.getBatchSize();

    std::vector<Color> batch_result(BATCH_SIZE);
    std::vector<Ray> batch_rays;
    std::vector<int> xs;
    std::vector<int> ys;
    int curr_sz = 0;

    Frame frame(vPixels, hPixels);

    for(int x=0;x<hPixels;x++) {
        for(int y=0;y<vPixels;y++) {
            Ray ray = createRay(x, y);

            batch_rays.push_back(ray);
            xs.push_back(x);
            ys.push_back(y);
            curr_sz++;

            if(curr_sz == BATCH_SIZE || (x == hPixels-1 && y == vPixels-1)) {
                scene.traceRayInBatch(batch_rays, batch_result);

                for(int i=0;i<curr_sz;i++) {
                    frame.setPixel(xs[i], (vPixels - ys[i] - 1), Pixel(batch_result[i]));
                }

                xs.clear();
                ys.clear();
                batch_rays.clear();

                curr_sz = 0;
            }
        }
    }

    return frame;
}

#endif
