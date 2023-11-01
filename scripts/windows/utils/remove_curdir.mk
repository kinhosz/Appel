PREFIX := $(patsubst %$(CHANGELOG),%,$(shell dir /b/s $(CHANGELOG)))

EMPTY := 
SPACE := $(EMPTY) $(EMPTY)

define split_cpp
$(subst .cpp_,.cpp$(SPACE),$1)
endef

define split_cu
$(subst .cu_,.cu$(SPACE),$1)
endef

define split_o
$(subst .o_,.o$(SPACE),$1)
endef

define whitespace_underline
$(call split_o,$(call split_cu,$(call split_cpp,$(subst $(SPACE),_,$1))))
endef

define remove_curdir
$(subst $(call whitespace_underline,$(PREFIX)),,$(call whitespace_underline,$1))
endef