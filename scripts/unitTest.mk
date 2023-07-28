# make unit FNAME=./../NAME_OF_FILE.cpp
unit:
	$(call compile,$(FNAME),tmp)
	$(call print_message,$(FNAME))
	$(call run,tmp)
