MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))
ARG_DIR := $(shell basename $(CURDIR))

test:
	earthly $(MAKEFILE_DIR)+test --DIR=$(ARG_DIR)

bench:
	earthly $(MAKEFILE_DIR)+bench --DIR=$(ARG_DIR)

bash:
	earthly $(MAKEFILE_DIR)+bash --DIR=$(ARG_DIR)
