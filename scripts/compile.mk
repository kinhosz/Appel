define compile
	@$(CC) -I $(INCLUDE_GEOMETRY) $1 $(GEOMETRY_CPP) -o $(BIN)/$2 $(CFLAGS)
endef
