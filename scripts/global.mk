CC := g++
CFLAGS := -Wall -Werror -O2 -std=c++17
INCLUDE := include
LIB_CPP := $(wildcard src/**/*.cpp)
LIB_SO := -Llib/SFML
LIBS_SO := -lsfml-audio -lsfml-graphics -lsfml-network -lsfml-system -lsfml-window

PREFIX_RUNNER_LD := LD_LIBRARY_PATH=lib/SFML

SEM_VER := include/version.hpp
README := README.md
CHANGELOG := CHANGELOG.md

BIN := bin
TEST := tests

PR_DESCRIPTION := Description here
