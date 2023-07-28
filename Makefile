include scripts/global.mk

ifeq ($(OS),Windows_NT)
	include scripts/windows.mk
else ifeq ($(shell uname), Linux)
	include scripts/linux.mk
endif

include scripts/compile.mk
include scripts/run.mk
include scripts/tests.mk
include scripts/unitTest.mk
