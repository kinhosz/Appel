PREFIX_RUNNER := ./
MKDIR := mkdir -p

build_bin:
	@$(MKDIR) $(BIN)
	@$(MKDIR) $(OBJ)

.PHONY: build_bin
