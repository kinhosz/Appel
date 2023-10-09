# get all *.cpp inside src/
CPP_SRCS := $(shell find $(SRC) -name '*.cpp')
CPP_OBJS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%, $(CPP_SRCS))

compile_cpp: build_bin $(CPP_OBJS)

# build .o files from .cpp
$(OBJ)/%: $(SRC)/%.cpp
	@echo $(GPU_MACROS)
	@$(CPP_COMPILER) -c -I $(INCLUDE) $< -o \
		$(OBJ)/$(subst /,_,$(subst $(OBJ),,$@)).o $(CPP_FLAGS) $(GPU_MACROS)
