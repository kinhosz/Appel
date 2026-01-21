include scripts/linux/dependencies/sfml.mk
include scripts/linux/dependencies/stb.mk

install:
	$(call sfml_install)
	$(call stb_install)
