ifeq ($(OS),Windows_NT)
	include scripts/win.mk
else
	echo Sistema nao suportado
endif
