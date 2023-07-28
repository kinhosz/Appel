CC := g++
CFLAGS := -O2 -Wall -Wextra
INCLUDE_GEOMETRY := include/geometry
GEOMETRY_CPP := src/geometry/*.cpp

BIN := bin/

# put here the source code name ex: mycode.cpp
PATH_CODE := tests/geometry/
MAIN_CODE_NAME := vetor.cpp
EXE_CODE_NAME := vetor

compile_all:
	$(CC) -I $(INCLUDE_GEOMETRY) $(PATH_CODE)$(MAIN_CODE_NAME) $(GEOMETRY_CPP) -o $(BIN)$(EXE_CODE_NAME) $(CFLAGS)

run:
	$(EXE_CODE_NAME)
