STB_H := https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
INCLUDE_STB := include/STB/

define stb_download
	@curl -LO $(STB_H) 2>/dev/null
endef

define stb_clear_destin_dir
	@rm -rf $(INCLUDE_STB)
endef

define stb_move_lib
	@mkdir -p include/STB && mv -f stb_image.h $(INCLUDE_STB)
endef

define stb_install
	@echo ------------------------------
	@echo Installing STB
	$(call stb_download)
	$(call stb_clear_destin_dir)
	$(call stb_move_lib)
	@echo OK
endef
