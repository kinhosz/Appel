#ifndef ENTITY_SCENE_H
#define ENTITY_SCENE_H

#include <map>
#include <entity/light.h>
#include <geometry/box.h>
#include <graphic/color.h>

class Scene {
private:
    std::map<int, Light> lights;
    int lightsCurrentIndex;
    Color environmentColor;
    std::map<int, Box> objects;
    int objectsCurrentIndex;
public:
    Scene();
    Scene(const Color& environmentColor);
    Scene(const std::map<int, Light>& lights, const Color& environmentColor, const std::map<int, Box>& objects);

    std::map<int, Light> getLights() const;
    Color getEnvironmentColor() const;
    std::map<int, Box> getObjects() const;

    int addLight(const Light& light);
    void removeLight(int index);
    void setEnvironmentColor(const Color& environmentColor);
    int addObject(const Box& object);
    void removeObject(int index);
};

#endif