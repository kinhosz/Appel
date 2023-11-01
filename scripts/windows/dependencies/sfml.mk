SFML_VERSION := 2.6.0
SFML_TAR_NAME := SFML-2.6.0-windows-gcc-13.1.0-mingw-64-bit.zip
SFML_BIN := https://github.com/SFML/SFML/releases/download/2.6.0/$(SFML_TAR_NAME)

LIB_SFML := lib\SFML
INCLUDE_SFML := include\SFML

define sfml_download
	@curl -LO $(SFML_BIN) 2>nul
endef

define sfml_unzip
	@tar -xzf $(SFML_TAR_NAME)
endef

define sfml_clear_destin_dir
	@if exist "$(INCLUDE_SFML)" rmdir /s /q $(INCLUDE_SFML)
	@if exist "$(LIB_SFML)" rmdir /s /q $(LIB_SFML)
endef

define sfml_move_lib
	@mkdir $(INCLUDE_SFML) && xcopy /E /Y SFML-2.6.0\include\SFML\* $(INCLUDE_SFML) 1>nul 2>nul
	@mkdir $(LIB_SFML) && xcopy /E /Y SFML-2.6.0\lib\* $(LIB_SFML) 1>nul 2>nul
endef

define sfml_clear_temp_files
	@del $(SFML_TAR_NAME)
	@rmdir /s /q SFML-2.6.0
endef

define sfml_install
	@echo ------------------------------
	@echo Installing SFML $(SFML_VERSION)
	$(call sfml_download)
	$(call sfml_unzip)
	$(call sfml_clear_destin_dir)
	$(call sfml_move_lib)
	$(call sfml_clear_temp_files)
	@echo OK
endef
