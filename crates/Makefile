MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))
ARG_DIR := $(shell basename $(CURDIR))

test:
	earthly $(MAKEFILE_DIR)+test --DIR=$(ARG_DIR)

publish-dry-run:
	earthly $(MAKEFILE_DIR)+publish --DIR=$(ARG_DIR)

publish:
	earthly --push $(MAKEFILE_DIR)+publish --DIR=$(ARG_DIR) --CARGO_REGISTRY_TOKEN=$(CARGO_REGISTRY_TOKEN)
	
bash:
	earthly $(MAKEFILE_DIR)+bash --DIR=$(ARG_DIR)
