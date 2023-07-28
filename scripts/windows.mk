PREFIX_RUNNER :=
MKDIR := mkdir

build_bin:
	@if not exist "$(BIN)" $(MKDIR) $(BIN)

.PHONY: build_bin