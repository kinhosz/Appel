#include <entity/scene.h>
#include <entity/light.h>
#include <geometry/point.h>
#include <graphic/color.h>
#include <vector>
#include <assert.h>
using namespace std;

int main() {
    Scene scene = Scene();
    assert(scene.getLights() == vector<Light>());
    assert(scene.getEnvironmentColor() == Color());

    vector<Light> lights = vector<Light>();
    lights.push_back(Light(Point(-3.00, 1.25, 0.30), Color(0.10, 0.80, 0.30)));
    lights.push_back(Light(Point(0.00, 1.00, 0.00), Color(1.00, 1.00, 1.00)));
    lights.push_back(Light(Point(2.00, -1.00, 0.70), Color(0.10, 0.68, 0.95)));
    scene.setLights(lights);
    assert(scene.getLights() == lights);

    Color environmentColor = Color(0.55, 0.78, 0.49);
    scene.setEnvironmentColor(environmentColor);
    assert(scene.getEnvironmentColor() == environmentColor);
}