#include <Appel/graphic/camera.h>
#include <Appel/geometry/point.h>

int main(){
    Point point(0, 0, 0);
    Point focus(1, 0, 0);
    int vPixels = 10;
    int hPixels = 20;
    double dist = 100;

    Camera camera1(point, focus, vPixels, hPixels);
    Camera camera2(point, focus, vPixels, hPixels, dist);

    camera1.zoom(100);
    
    Point movement(-1, -1, -1);
    camera1.move(movement);
}
