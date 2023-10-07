include scripts/global.mk

ifeq ($(OS),Windows_NT)
	include scripts/windows.mk
else ifeq ($(shell uname), Linux)
	include scripts/linux.mk
endif

include scripts/install.mk
include scripts/compile.mk
include scripts/run.mk
include scripts/tests.mk
include scripts/unitTest.mk
include scripts/changelog.mk
include scripts/semVer.mk
include scripts/compile_all.mk
