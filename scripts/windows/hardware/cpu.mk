GPU_MACROS := -DAPPEL_GPU_DISABLED

compile_cuda:
	@echo GPU DISABLED

define compile
	@$(CPP_COMPILER) -c -I$(INCLUDE) $1 -o $(BIN)\tmp.o $(CPP_FLAGS) $(GPU_MACROS) -DSFML_STATIC
	@$(CPP_COMPILER) -Llib\SFML $(BIN)\tmp.o $(call remove_curdir,$(shell dir /b/s $(OBJ)\*.o)) \
		-o $(BIN)\$(subst \,_,$2).exe \
		-lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lwinmm -lopengl32 -lgdi32 -lfreetype -lsfml-main
endef
