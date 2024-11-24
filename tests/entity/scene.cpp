#include <entity/scene.h>
#include <entity/light.h>
#include <geometry/point.h>
#include <graphic/color.h>
#include <vector>
#include <assert.h>
using namespace std;

int main() {
    Scene scene1, scene2(Color(1.0, 1.0, 0.9)), scene3(Color(0.5, 0.0, 0.4));

    scene3.addLight(Light(Point(0, 0, 0), Color(0.1, 0.2, 0.3)));
    scene3.addLight(Light(Point(1, 1, 1), Color(0.2, 0.3, 0.4)));
    scene3.addLight(Light(Point(2, 2, 2), Color(0.3, 0.4, 0.5)));

    Sphere sphere(Point(), 0, Color());

    sphere.setPhongValues(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    scene3.addObject(sphere);

    sphere.setPhongValues(0.2, 0.3, 0.4, 0.5, 0.6, 0.7);
    scene3.addObject(sphere);

    sphere.setPhongValues(0.3, 0.4, 0.5, 0.6, 0.7, 0.8);
    scene3.addObject(sphere);

    assert(scene1.getLights().size() == 0);
    assert(scene1.getEnvironmentColor() == Color(0.0, 0.0, 0.0));
    assert(scene2.getLights().size() == 0);
    assert(scene2.getEnvironmentColor() == Color(1.0, 1.0, 0.9));
    assert(scene3.getLights().size() == 3);
    assert(scene3.getLights()[0] == Light(Point(0, 0, 0), Color(0.1, 0.2, 0.3)));
    assert(scene3.getLights()[1] == Light(Point(1, 1, 1), Color(0.2, 0.3, 0.4)));
    assert(scene3.getLights()[2] == Light(Point(2, 2, 2), Color(0.3, 0.4, 0.5)));
    assert(scene3.getEnvironmentColor() == Color(0.5, 0.0, 0.4));

    Light light1(Point(0, 0, 0), Color(0.1, 0.2, 0.3));
    Light light2(Point(1, 1, 1), Color(0.2, 0.3, 0.4));

    Sphere box1(Point(), 0.0, Color());
    Sphere box2(Point(), 0.0, Color());

    box1.setPhongValues(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    box2.setPhongValues(0.2, 0.3, 0.4, 0.5, 0.6, 0.7);

    scene1.addLight(light1);
    scene1.addLight(light2);
    scene1.setEnvironmentColor(Color(1.0, 0.8, 0.0));
    scene1.addObject(box1);
    scene1.addObject(box2);

    assert(scene1.getLights().size() == 2);
    assert(scene1.getLights()[0] == light1);
    assert(scene1.getLights()[1] == light2);
    assert(scene1.getEnvironmentColor() == Color(1.0, 0.8, 0.0));

    scene1.removeLight(1);
    scene1.removeObject(0);

    assert(scene1.getLights().size() == 1);
    assert(scene1.getLights()[0] == light1);
}