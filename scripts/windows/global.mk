EGPU := 1

CPP_COMPILER := g++
CPP_FLAGS := -Wall -Werror -O2 -std=gnu++20

CUDA_COMPILER := nvcc
CUDA_FLAGS := -Xcompiler -Wall -Xcompiler -Werror -Xcompiler -O2 -std=c++17

INCLUDE := $(PREFIX)include
LIB_CPP = $(call remove_curdir,$(shell dir /b/s $(SRC)\*.cpp))
LIB_CU = $(call remove_curdir,$(shell dir /b/s $(SRC)\*.cu))
LINK_FLAGS = -Llib\SFML $(LIBS_SO)
LIBS_SO := -lsfml-graphics-s -lsfml-window-s -lsfml-system-s
# -lcudart ?
PREFIX_RUNNER_LD := LD_LIBRARY_PATH=lib\SFML

README := README.md
CHANGELOG := CHANGELOG.md

SRC := src
BIN := bin
TEST := tests
OBJ := $(BIN)\obj

PREFIX_RUNNER :=
MKDIR := mkdir

print:
	@echo $(LIB_CPP)
	@echo $(LIB_CU)
	@echo --------

build_bin:
	@if not exist "$(BIN)" $(MKDIR) $(BIN)
	@if not exist "$(OBJ)" $(MKDIR) $(OBJ)

.PHONY: build_bin
