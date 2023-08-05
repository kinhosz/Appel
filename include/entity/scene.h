#ifndef ENTITY_SCENE_H
#define ENTITY_SCENE_H

#include <vector>
#include <entity/light.h>
#include <graphic/color.h>

class Scene {
private:
    std::vector<Light> lights;
    Color environmentColor; // QUESTION: Should this be a Color or a Pixel?
    // TO DO: Add objects
public:
    Scene();
    Scene(std::vector<Light> lights, Color environmentColor);

    std::vector<Light> getLights() const;
    Color getEnvironmentColor() const;

    void setLights(std::vector<Light> lights);
    void setEnvironmentColor(Color environmentColor);

    //TO DO: Add objects getters and setters
    //TO DO: Add intersection method
};

#endif