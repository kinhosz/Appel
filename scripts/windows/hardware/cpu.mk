GPU_MACROS := -DAPPEL_GPU_DISABLED

compile_cuda:
	@echo GPU DISABLED

define compile
	$(CPP_COMPILER) -c -I$(INCLUDE) $1 -o $(BIN)\tmp.o $(CPP_FLAGS) $(GPU_MACROS)
	$(CPP_COMPILER) -o $(BIN)\$(subst \,_,$2).exe $(BIN)\tmp.o $(call remove_curdir,$(shell dir /b/s $(OBJ)\*.o)) $(LINK_FLAGS)
endef
