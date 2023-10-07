define compile
	@$(CC) $(LIB_SO) -I $(INCLUDE) $1 $(LIB_CU) $(LIB_CPP) -o $(BIN)/$2 $(LIBS_SO) $(CFLAGS)
endef

