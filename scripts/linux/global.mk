EGPU := 1

CPP_COMPILER := g++
CPP_FLAGS := -Wall -Werror -O2 -std=c++17

CUDA_COMPILER := nvcc
CUDA_FLAGS := -Xcompiler -Wall -Xcompiler -Werror -Xcompiler -O2 -std=c++17

INCLUDE := include
LIB_CPP := $(shell find src -type f -name "*.cpp")
LIB_CU := $(shell find src -type f -name "*.cu")
LIB_SO := -Llib/SFML
LIBS_SO := -lsfml-audio -lsfml-graphics -lsfml-network -lsfml-system -lsfml-window
# -lcudart ?
PREFIX_RUNNER_LD := LD_LIBRARY_PATH=lib/SFML

SEM_VER := include/version.hpp
README := README.md
CHANGELOG := CHANGELOG.md

SRC := src
BIN := bin
TEST := tests
OBJ := $(BIN)/obj

PR_DESCRIPTION := Description here

PREFIX_RUNNER := ./
MKDIR := mkdir -p

build_bin:
	@$(MKDIR) $(BIN)
	@$(MKDIR) $(OBJ)

.PHONY: build_bin

