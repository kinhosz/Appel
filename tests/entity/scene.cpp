#include <entity/scene.h>
#include <entity/light.h>
#include <geometry/point.h>
#include <graphic/color.h>
#include <vector>
#include <assert.h>
using namespace std;

int main() {
    Scene scene1, scene2(Color(10, 80, 155)), scene3(
        map<int, Light>({
            {0, Light(Point(0, 0, 0), Color(15, 80, 255))},
            {1, Light(Point(1, 1, 1), Color(35, 255, 0))},
            {2, Light(Point(2, 2, 2), Color(50, 0, 100))}
        }),
        Color(255, 0, 100),
        map<int, Box>({
            {0, Box(Point(0, 0, 0), Point(1, 1, 1), Color(10, 255, 15))},
            {1, Box(Point(1, 1, 1), Point(2, 2, 2), Color(15, 0, 0))},
            {2, Box(Point(2, 2, 2), Point(3, 3, 3), Color(255, 10, 255))}
        })
    );

    assert(scene1.getLights().size() == 0);
    assert(scene1.getEnvironmentColor() == Color(0, 0, 0));
    assert(scene1.getObjects().size() == 0);
    assert(scene2.getLights().size() == 0);
    assert(scene2.getEnvironmentColor() == Color(10, 80, 155));
    assert(scene2.getObjects().size() == 0);
    assert(scene3.getLights().size() == 3);
    assert(scene3.getLights()[0] == Light(Point(0, 0, 0), Color(15, 80, 255)));
    assert(scene3.getLights()[1] == Light(Point(1, 1, 1), Color(35, 255, 0)));
    assert(scene3.getLights()[2] == Light(Point(2, 2, 2), Color(50, 0, 100)));
    assert(scene3.getEnvironmentColor() == Color(255, 0, 100));
    assert(scene3.getObjects().size() == 3);
    assert(scene3.getObjects()[0] == Box(Point(0, 0, 0), Point(1, 1, 1), Color(10, 255, 15)));
    assert(scene3.getObjects()[1] == Box(Point(1, 1, 1), Point(2, 2, 2), Color(15, 0, 0)));
    assert(scene3.getObjects()[2] == Box(Point(2, 2, 2), Point(3, 3, 3), Color(255, 10, 255)));

    Light light1(Point(0, 0, 0), Color(15, 80, 0));
    Light light2(Point(1, 1, 1), Color(35, 255, 255));
    Light light3(Point(2, 2, 2), Color(50, 100, 255));

    Box box1(Point(0, 0, 0), Point(1, 1, 1), Color(255, 0, 0));
    Box box2(Point(1, 1, 1), Point(2, 2, 2), Color(0, 255, 0));
    Box box3(Point(2, 2, 2), Point(3, 3, 3), Color(0, 0, 255));

    scene1.addLight(light1);
    scene1.addLight(light2);
    scene1.addLight(light3);
    scene1.setEnvironmentColor(Color(255, 255, 255));
    scene1.addObject(box1);
    scene1.addObject(box2);
    scene1.addObject(box3);

    assert(scene1.getLights().size() == 3);
    assert(scene1.getLights()[0] == light1);
    assert(scene1.getLights()[1] == light2);
    assert(scene1.getLights()[2] == light3);
    assert(scene1.getEnvironmentColor() == Color(255, 255, 255));
    assert(scene1.getObjects().size() == 3);
    assert(scene1.getObjects()[0] == box1);
    assert(scene1.getObjects()[1] == box2);
    assert(scene1.getObjects()[2] == box3);

    scene1.removeLight(1);
    scene1.removeObject(0);

    assert(scene1.getLights().size() == 2);
    assert(scene1.getLights()[0] == light1);
    assert(scene1.getLights()[2] == light3);
    assert(scene1.getObjects().size() == 2);
    assert(scene1.getObjects()[1] == box2);
    assert(scene1.getObjects()[2] == box3);
}