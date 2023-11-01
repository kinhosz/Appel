# get all *.cpp inside src/
CPP_SRCS = $(call remove_curdir,$(shell dir /b/s $(SRC)\*.cpp))
CPP_OBJS = $(patsubst $(SRC)\\%.cpp, $(OBJ)\\%, $(CPP_SRCS))

compile_cpp: build_bin $(CPP_OBJS)

# build .o files from .cpp
$(OBJ)\\%: $(SRC)\%.cpp
	@$(CPP_COMPILER) -c -I $(INCLUDE) $< -o \
		$(OBJ)\$(subst \,_,$(subst $(OBJ),,$@)).o $(CPP_FLAGS) $(GPU_MACROS) -DSFML_STATIC
