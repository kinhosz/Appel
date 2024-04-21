# get all *.cpp inside tests/
TESTS := $(call remove_curdir,$(shell dir /b/s $(TEST)\*.cpp))
BINS := $(patsubst $(TEST)\\%.cpp, $(BIN)\\%, $(TESTS))

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
$(BIN)\\%: $(TEST)\%.cpp
	$(call compile,$<,$@)
	$(call print_message,$(subst /,_,$@))
	$(call run,$(subst \,_,$@).exe)
