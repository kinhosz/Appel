#include <iostream>
#include <vector>
#include <geometry/point.h>
#include <geometry/triangle.h>
#include <entity/triangularMesh.h>
#include <entity/scene.h>
#include <graphic/camera.h>
#include <assert.h>
#include <graphic/utils.h>
#include <SFML/Graphics.hpp>
#include <math.h>
using namespace std;

#define WIDTH 1280
#define HEIGHT 720
#define FPS 30

// maps
std::map<int, Triangle> m_triangles;
int idx_triangle = 0;
std::map<int, Point> m_points;
int idx_point = 0;
std::map<int, Color> m_colors;
int idx_color = 0;

// backlog
std::vector<std::string> backlog;
std::string current_command = "";

// material
int mesh_index = -1;

void cli(
  const sf::Font &font,
  sf::RenderWindow &window
) {
  float x = 770.f, y = 400;

  sf::RectangleShape panel;
  panel.setSize(sf::Vector2f(500.f, 300.f));
  panel.setPosition(x, y);
  panel.setFillColor(sf::Color(0, 0, 80));
  panel.setOutlineThickness(2.f);
  panel.setOutlineColor(sf::Color::White);
  window.draw(panel);

  for(const auto& line : backlog) {
    sf::Text text(line, font, 10);
    text.setPosition(x, y);
    window.draw(text);
    y += 18.f;
  }
  std::string cli_cmd = "> " + current_command;
  sf::Text text(cli_cmd, font, 10);
  text.setPosition(x, y);
  text.setFillColor(sf::Color::Green);
  window.draw(text);
}

std::string execute_cmd(std::string cmd) {
  std::vector<std::string> args;
  std::string arg = "";

  for(int i=0;i<(int)cmd.size();i++) {
    if(cmd[i] == ' ') {
      args.push_back(arg);
      arg = "";
    } else {
      arg += cmd[i];
    }
  }
  args.push_back(arg);
  arg = "";
  std::string response = "invalid command";

  if(args.size() == 0) return response;

  auto is_double = [&](std::string m) {
    try { std::stod(m); }
    catch (const std::exception&) { return false; }
    return true;
  };

  auto is_int = [&](std::string m) {
    try { std::stoi(m); }
    catch (const std::exception&) { return false; }
    return true;
  };

  if(args[0] == "set") {
    if(args[1] == "point" && args.size() == 5) {
      if(is_double(args[2]) && is_double(args[3]) && is_double(args[4])) {
        m_points[idx_point] = Point(
          std::stod(args[2]),
          std::stod(args[3]),
          std::stod(args[4])
        );
        response = "Point added: " + std::to_string(idx_point);
        idx_point++;
      }
    } else if(args[1] == "color" && args.size() == 5) {
      if(is_int(args[2]) && is_int(args[3]) && is_int(args[4])) {
        m_colors[idx_color] = Color(
          std::stoi(args[2]),
          std::stoi(args[3]),
          std::stoi(args[4])
        );
        response = "Color added: " + std::to_string(idx_color);
        idx_color++;
      }
    } else if(args[1] == "triangle" && args.size() == 6) {
      if(is_int(args[2]) && is_int(args[3]) && is_int(args[4]) && is_int(args[5])) {
        int i_0 = std::stoi(args[2]);
        int i_1 = std::stoi(args[3]);
        int i_2 = std::stoi(args[4]);
        int i_c = std::stoi(args[5]);

        if(m_points.find(i_0) == m_points.end()) {
          return "Point " + std::to_string(i_0) + " doesnt exist";
        }
        if(m_points.find(i_1) == m_points.end()) {
          return "Point " + std::to_string(i_1) + " doesnt exist";
        }
        if(m_points.find(i_2) == m_points.end()) {
          return "Point " + std::to_string(i_2) + " doesnt exist";
        }
        if(m_colors.find(i_c) == m_colors.end()) {
          return "Color " + std::to_string(i_c) + " doesnt exist";
        }

        Point p0 = m_points[i_0], p1 = m_points[i_1], p2 = m_points[i_2];
        Color c = m_colors[i_c];
        
        m_triangles[idx_triangle] = Triangle(p0, p1, p2, c);
        response = "Triangle added: " + std::to_string(idx_triangle);
        idx_triangle++;
      }
    }
  } else if(args[0] == "GET") {

  } else if(args[0] == "DEL") {

  }

  return response;
}

void reload_material(Scene &scene) {
  if(mesh_index != -1) scene.removeObject(mesh_index);

  std::vector<Triangle> triangles;
  for(auto &p: m_triangles) {
    triangles.push_back(p.second);
  }

  TriangularMesh mesh(triangles);
  mesh_index = scene.addObject(mesh);
}

int main() {
  Scene scene(1);
  Camera camera(
    Point(10, 0, 10),
    Point(0, 0, 0),
    HEIGHT,
    WIDTH
  );

  sf::Font font;
  font.loadFromFile("assets/fonts/PressStart2P-Regular.ttf");

  Frame frame = camera.take(scene, false);

  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mini Blender");

  bool show_cli = false;

  while(window.isOpen()) {
    sf::Event event;
    while(window.pollEvent(event)) {
      if(event.type == sf::Event::Closed) window.close();
      if(event.type == sf::Event::TextEntered) {
        // open CLI
        if(!show_cli) {
          if(event.text.unicode == 'T' || event.text.unicode == 't') show_cli = true;
        } else {
          // Backspace
          if(event.text.unicode == 8) {
            if(!current_command.empty()) current_command.pop_back();
          }
          // Enter
          else if(event.text.unicode == 13) {
            if(current_command == "quit") {
              show_cli = false;
            } else{
              backlog.push_back(current_command);
              std::string response = execute_cmd(current_command);
              backlog.push_back(response);
              while(backlog.size() > 15) backlog.erase(backlog.begin());
              reload_material(scene);
            }
            current_command = "";
          }
          // Caracteres ASCII visíveis
          else if(event.text.unicode < 128 && current_command.size() < 20) {
            current_command += static_cast<char>(event.text.unicode);
          }
        }
      }
    }

    window.clear();
    /************************************************/
    sf::Image image;
    image.create(WIDTH, HEIGHT, sf::Color::White);

    for (int x = 0; x < frame.horizontal(); ++x) {
      for (int y = 0; y < frame.vertical(); ++y) {
        sf::Color color(
          frame.getPixel(x, y).getRed(),
          frame.getPixel(x, y).getGreen(),
          frame.getPixel(x, y).getBlue()
        );
        image.setPixel(x, y, color);
      }
    }

    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite(texture);

    window.draw(sprite);
    /************************************************/
    if(show_cli) cli(font, window);
    /************************************************/
    frame = camera.take(scene, false);

    window.display();
  }

  return 0;
}
