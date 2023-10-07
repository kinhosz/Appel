# get all *.cpp inside tests/
TESTS := $(shell find $(TEST) -name '*.cpp')
BINS := $(patsubst $(TEST)/%.cpp, $(BIN)/%, $(TESTS))

# make tests
tests: build_bin compile_cpp compile_cuda $(BINS)

define print_message
	@echo ------------------------
	@echo $(1)
endef

# compile all *.cpp inside tests/
# $@ will be replaced by the target
# $< will be replaced by the dependecy
# compile & run code
$(BIN)/%: $(TEST)/%.cpp
	@$(CPP_COMPILER) -c -I $(INCLUDE) $< -o $(BIN)/tmp.o $(CPP_FLAGS)
	@$(CUDA_COMPILER) $(LIB_SO) $(BIN)/tmp.o $(shell find $(OBJ) -name '*.o') \
		 -o $(BIN)/$(subst /,_,$@).exe $(LIBS_SO)
	$(call print_message,$(subst /,_,$@))
	$(call run,$(subst /,_,$@).exe)
