define run
	@$(PREFIX_RUNNER_LD) nvprof $(PREFIX_RUNNER)$(BIN)/$1
endef
