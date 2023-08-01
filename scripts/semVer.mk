PREFIX_IN_FILE := \#define

MAJOR_IDENTIFIER = $(PREFIX_IN_FILE) APPEL_MAJOR
MINOR_IDENTIFIER = $(PREFIX_IN_FILE) APPEL_MINOR
PATCH_IDENTIFIER = $(PREFIX_IN_FILE) APPEL_PATCH

OLD_MAJOR := $(shell grep "$(MAJOR_IDENTIFIER)" $(SEM_VER))
OLD_MINOR := $(shell grep "$(MINOR_IDENTIFIER)" $(SEM_VER))
OLD_PATCH := $(shell grep "$(PATCH_IDENTIFIER)" $(SEM_VER))

NUMBER_MAJOR := $(subst $(MAJOR_IDENTIFIER) ,,$(OLD_MAJOR))
NUMBER_MINOR := $(subst $(MINOR_IDENTIFIER) ,,$(OLD_MINOR))
NUMBER_PATCH := $(subst $(PATCH_IDENTIFIER) ,,$(OLD_PATCH))

INCREMENTED_MAJOR := $(shell echo $$(( $(NUMBER_MAJOR) + 1 )))
INCREMENTED_MINOR := $(shell echo $$(( $(NUMBER_MINOR) + 1 )))
INCREMENTED_PATCH := $(shell echo $$(( $(NUMBER_PATCH) + 1 )))

NEW_MAJOR := $(MAJOR_IDENTIFIER) $(INCREMENTED_MAJOR)
NEW_MINOR := $(MINOR_IDENTIFIER) $(INCREMENTED_MINOR)
NEW_PATCH := $(PATCH_IDENTIFIER) $(INCREMENTED_PATCH)

RESETED_MINOR := $(MINOR_IDENTIFIER) 0
RESETED_PATCH := $(PATCH_IDENTIFIER) 0

define update_version
	sed -i -e 's/$(OLD_MAJOR)/$($1)/g' $(SEM_VER)
	sed -i -e 's/$(OLD_MINOR)/$($2)/g' $(SEM_VER)
	sed -i -e 's/$(OLD_PATCH)/$($3)/g' $(SEM_VER)
endef

define update_readme
	sed -i -e 's/\*\*Current Version\*\*: *.*.*/\*\*Current Version\*\*: $1.$2.$3/g' $(README)
endef

update_major:
	$(call update_version,NEW_MAJOR,RESETED_MINOR,RESETED_PATCH)
	$(call update_readme,$(INCREMENTED_MAJOR),0,0)

	$(call changelog_major,$(INCREMENTED_MAJOR))
	$(call changelog_minor,0)
	$(call changelog_patch,0,$(subst /,\/,$(PR_DESCRIPTION)))

update_minor:
	$(call update_version,OLD_MAJOR,NEW_MINOR,RESETED_PATCH)
	$(call update_readme,$(NUMBER_MAJOR),$(INCREMENTED_MINOR),0)

	$(call changelog_minor,$(INCREMENTED_MINOR))
	$(call changelog_patch,0,$(subst /,\/,$(PR_DESCRIPTION)))

update_patch:
	$(call update_version,OLD_MAJOR,OLD_MINOR,NEW_PATCH)
	$(call update_readme,$(NUMBER_MAJOR),$(NUMBER_MINOR),$(INCREMENTED_PATCH))

	$(call changelog_patch,$(INCREMENTED_PATCH),$(subst /,\/,$(PR_DESCRIPTION)))
