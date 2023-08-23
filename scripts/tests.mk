# get all *.cpp inside tests/
TESTS := $(shell find $(TEST) -name '*.cpp')
BINS := $(patsubst $(TEST)/%.cpp, $(BIN)/%, $(TESTS))

# make tests
tests: build_bin $(BINS)

define print_message
	@echo ------------------------
	@echo $(1)
endef

# compile all *.cpp inside tests/
# $@ will be replaced by the target
# $< will be replaced by the dependecy
# compile & run code
$(BIN)/%: $(TEST)/%.cpp
	$(call compile,$<,$(subst /,_,$@))
	$(call print_message,$(subst /,_,$@))
	$(call run,$(subst /,_,$@))
