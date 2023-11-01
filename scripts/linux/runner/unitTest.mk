# make unit FNAME=./../NAME_OF_FILE.cpp
unit: build_bin compile_cpp compile_cuda
	$(call compile,$(FNAME),example)
	$(call print_message,$(FNAME))
	$(call run,example.exe)
