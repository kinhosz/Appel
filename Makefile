include scripts/global.mk

ifeq ($(OS),Windows_NT)
include scripts/OS/windows.mk
else ifeq ($(shell uname), Linux)
include scripts/OS/linux.mk
endif

ifeq ($(EGPU), 1)
include scripts/hardware/gpu.mk
else
include scripts/hardware/cpu.mk
endif

include scripts/dependencies/install.mk

include scripts/runner/run.mk
include scripts/runner/tests.mk
include scripts/runner/unitTest.mk
include scripts/runner/build.mk

include scripts/versioning/changelog.mk
include scripts/versioning/semVer.mk
