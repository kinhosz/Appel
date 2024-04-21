# windows system
ifeq ($(OS),Windows_NT)

include scripts/windows/global.mk
include scripts/windows/utils/remove_curdir.mk
include scripts/windows/dependencies/install.mk
include scripts/windows/runner/tests.mk
include scripts/windows/runner/unitTest.mk
include scripts/windows/runner/build.mk
include scripts/windows/runner/clear.mk
include scripts/windows/runner/run.mk

ifeq ($(EGPU), 1)
include scripts/windows/hardware/gpu.mk
else
include scripts/windows/hardware/cpu.mk
endif

# linux system
else ifeq ($(shell uname), Linux)

include scripts/linux/global.mk
include scripts/linux/dependencies/install.mk
include scripts/linux/runner/run.mk
include scripts/linux/runner/tests.mk
include scripts/linux/runner/unitTest.mk
include scripts/linux/runner/build.mk
include scripts/linux/runner/clear.mk
include scripts/linux/versioning/changelog.mk
include scripts/linux/versioning/semVer.mk

ifeq ($(EGPU), 1)
include scripts/linux/hardware/gpu.mk
else
include scripts/linux/hardware/cpu.mk
endif

endif