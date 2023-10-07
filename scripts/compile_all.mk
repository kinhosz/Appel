# get all *.cpp inside src/
CPP_SRCS := $(shell find $(SRC) -name '*.cpp')
CPP_OBJS := $(patsubst $(SRC)/%.cpp, $(OBJ)/%, $(CPP_SRCS))

compile_cpp: build_bin $(CPP_OBJS)

# build .o files from .cpp
$(OBJ)/%: $(SRC)/%.cpp
	@$(CPP_COMPILER) -c -I $(INCLUDE) $< -o \
		$(OBJ)/$(subst /,_,$(subst $(OBJ),,$@)).o $(CPP_FLAGS)

# get all *.cu inside src/
CUDA_SRCS := $(shell find $(SRC) -name '*.cu')
CUDA_OBJS := $(patsubst $(SRC)/%.cu, $(OBJ)/%, $(CUDA_SRCS))

compile_cuda: build_bin $(CUDA_OBJS)

# build .o files from .cu
$(OBJ)/%: $(SRC)/%.cu
	@$(CUDA_COMPILER) -c -dc -I $(INCLUDE) $< -o \
		$(OBJ)/$(subst /,_,$(subst $(OBJ),,$@)).o $(CUDA_FLAGS)
