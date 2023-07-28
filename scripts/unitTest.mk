# make unit FNAME=./../NAME_OF_FILE.cpp
unit: build_bin
	$(call compile,$(FNAME),tmp)
	$(call print_message,$(FNAME))
	$(call run,tmp)
