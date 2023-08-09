PREFIX_RUNNER := ./
MKDIR := mkdir -p

build_bin:
	@$(MKDIR) $(BIN)

.PHONY: build_bin
