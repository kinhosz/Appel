CC := g++
CFLAGS := -Wall -Werror -O2 -std=c++17
INCLUDE := include
LIB_CPP := $(wildcard src/**/*.cpp)
SEM_VER := include/version.hpp
README := README.md
CHANGELOG := CHANGELOG.md

BIN := bin
TEST := tests

PR_DESCRIPTION := Description here
