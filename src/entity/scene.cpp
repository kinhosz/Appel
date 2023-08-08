#include <entity/scene.h>
#include <assert.h>

Scene::Scene() {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size() - 1;
    this->environmentColor = Color(0, 0, 0);
    this->objects = std::map<int, Box>();
    this->objectsCurrentIndex = this->objects.size() - 1;
}

Scene::Scene(const Color& environmentColor) {
    this->lights = std::map<int, Light>();
    this->lightsCurrentIndex = this->lights.size() - 1;
    this->environmentColor = environmentColor;
    this->objects = std::map<int, Box>();
    this->objectsCurrentIndex = this->objects.size() - 1;
}

Scene::Scene(const std::map<int, Light>& lights, const Color& environmentColor, const std::map<int, Box>& objects) {
    this->lights = lights;
    this->lightsCurrentIndex = this->lights.size() - 1;
    this->environmentColor = environmentColor;
    this->objects = objects;
    this->objectsCurrentIndex = this->objects.size() - 1;
}

std::map<int, Light> Scene::getLights() const {
    return this->lights;
}

Color Scene::getEnvironmentColor() const {
    return this->environmentColor;
}

std::map<int, Box> Scene::getObjects() const {
    return this->objects;
}

int Scene::addLight(const Light& light) {
    this->lightsCurrentIndex++;
    this->lights[lightsCurrentIndex] = light;
    return lightsCurrentIndex;
}

void Scene::removeLight(int index) {
    assert(this->lights.find(index) != this->lights.end());
    this->lights.erase(index);
}

void Scene::setEnvironmentColor(const Color& environmentColor) {
    this->environmentColor = environmentColor;
}

int Scene::addObject(const Box& object) {
    this->objectsCurrentIndex++;
    this->objects[objectsCurrentIndex] = object;
    return objectsCurrentIndex;
}

void Scene::removeObject(int index) {
    assert(this->objects.find(index) != this->objects.end());
    this->objects.erase(index);
}