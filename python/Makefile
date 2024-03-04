MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(dir $(MAKEFILE_PATH))
ARG_DIR := $(shell basename $(CURDIR))

pytest:
	earthly $(MAKEFILE_DIR)+pytest --DIR=$(ARG_DIR)

bash:
	earthly $(MAKEFILE_DIR)+bash --DIR=$(ARG_DIR)
