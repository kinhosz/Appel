GPU_MACROS := -DAPPEL_GPU

# get all *.cu inside src/
CUDA_SRCS := $(shell find $(SRC) -name '*.cu')
CUDA_OBJS := $(patsubst $(SRC)/%.cu, $(OBJ)/%, $(CUDA_SRCS))

compile_cuda: build_bin $(CUDA_OBJS)
	@echo "gpu ok"

# build .o files from .cu
$(OBJ)/%: $(SRC)/%.cu
	@$(CUDA_COMPILER) -c -dc -I $(INCLUDE) $< -o \
		$(OBJ)/$(subst /,_,$(subst $(OBJ),,$@)).o $(CUDA_FLAGS)

define compile
	@$(CPP_COMPILER) -c -I $(INCLUDE) $1 -o $(BIN)/tmp.o $(CPP_FLAGS) -DAPPEL_GPU
	@$(CUDA_COMPILER) $(LIB_SO) $(BIN)/tmp.o $(shell find $(OBJ) -name '*.o') \
		-o $(BIN)/$(subst /,_,$2).exe $(LIBS_SO)
endef
