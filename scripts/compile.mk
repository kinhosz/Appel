define compile
	@$(CC) $(LIB_SO) -I $(INCLUDE) $1 $(LIB_CPP) $(LIB_CU) -o $(BIN)/$2 $(LIBS_SO) $(CFLAGS)
endef
