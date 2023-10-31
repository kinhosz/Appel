define run
	@set PATH="$(PREFIX)lib\SFML;"%PATH%
	$(PREFIX_RUNNER)$(BIN)\$1 $(LINK_FLAGS)
endef
