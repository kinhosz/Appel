#include <entity/scene.h>

Scene::Scene() : lights(std::vector<Light>()), environmentColor(Color()) {}

Scene::Scene(std::vector<Light> lights, Color environmentColor) : lights(lights), environmentColor(environmentColor) {}

std::vector<Light> Scene::getLights() const {
    return this->lights;
}

Color Scene::getEnvironmentColor() const {
    return this->environmentColor;
}

void Scene::setLights(std::vector<Light> lights) {
    this->lights = lights;
}

void Scene::setEnvironmentColor(Color environmentColor) {
    this->environmentColor = environmentColor;
}