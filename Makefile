ifeq ($(OS),Windows_NT)
	include scripts/win.mk
else
	echo Sistema nao suportado
endif

include scripts/compile.mk
include scripts/run.mk
include scripts/tests.mk
include scripts/unitTest.mk
