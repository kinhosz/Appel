define compile
	$(CC) -I $(INCLUDE) $1 $(LIB_CPP) -o $(BIN)/$2 $(CFLAGS)
endef
