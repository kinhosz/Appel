#1. baixar arquivo: curl -LO https://github.com/SFML/SFML/releases/download/2.6.0/SFML-2.6.0-linux-gcc-64-bit.tar.gz
#2. decompactar: tar -xzf SFML-2.6.0-linux-gcc-64-bit.tar.gz
#3. movendo include: mkdir -p include/SFML-teste && mv SFML-2.6.0/include/SFML/* include/SFML-teste/
#4. movendo lib: mkdir -p lib/SFML-teste && mv SFML-2.6.0/lib/* lib/SFML-teste/
#5. rm -r SFML-2.6.0-linux-gcc-64-bit.tar.gz
#6. rm -r SFML-2.6.0

SFML_TAR_NAME := SFML-2.6.0-linux-gcc-64-bit.tar.gz
SFML_BIN := https://github.com/SFML/SFML/releases/download/2.6.0/$(SFML_TAR_NAME)

define sfml_install
	curl -LO $(SFML_BIN)
	tar -xzf $(SFML_TAR_NAME)
	mkdir -p include/SFML && mv SFML-2.6.0/include/SFML/* include/SFML/
	mkdir -p lib/SFML && mv SFML-2.6.0/lib/* lib/SFML/
	rm -r SFML-2.6.0-linux-gcc-64-bit.tar.gz
	rm -r SFML-2.6.0
endef
