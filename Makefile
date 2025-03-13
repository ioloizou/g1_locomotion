SHELL := /bin/bash

# Define workspace directories
WS_DIR := /home/forest_ws
SRC_DIR := $(WS_DIR)/src
LOCO_DIR := $(SRC_DIR)/g1_locomotion
BUILD_DIR := $(WS_DIR)/build
INSTALL_DIR := $(WS_DIR)/install

# Get all valid ROS packages inside g1_locomotion (only directories with package.xml)
LOCO_PACKAGES := $(shell find $(LOCO_DIR) -maxdepth 1 -mindepth 1 -type d -exec test -f {}/package.xml \; -print | xargs -I{} basename {})

# Default target: Build all g1_locomotion packages
.PHONY: all
all: build

# Build all g1_locomotion packages using separate Make targets for each package,
# which enables proper dependency tracking and parallel builds.
.PHONY: build $(LOCO_PACKAGES)
build: $(LOCO_PACKAGES)
$(LOCO_PACKAGES):
	@echo "Building package $@"
	@mkdir -p "$(BUILD_DIR)/$@"
	@cd "$(BUILD_DIR)/$@" && \
		source /opt/ros/noetic/setup.bash && \
		source $(WS_DIR)/setup.bash && \
		cmake -DCMAKE_INSTALL_PREFIX="$(INSTALL_DIR)" \
			  -DCMAKE_BUILD_TYPE=Release \
			  "$(LOCO_DIR)/$@" && \
		$(MAKE) -j$(nproc) && $(MAKE) install

# Build a specific package inside g1_locomotion
.PHONY: pkg
pkg:
ifndef PKG
	$(error PKG is not set. Please use: make pkg PKG=<package_name>)
endif
	@echo "Building package $(PKG) step-by-step:"
	@mkdir -p "$(BUILD_DIR)/$(PKG)"
	@cd "$(BUILD_DIR)/$(PKG)" && \
		source /opt/ros/noetic/setup.bash && \
		source $(WS_DIR)/setup.bash && \
		cmake -DCMAKE_INSTALL_PREFIX="$(INSTALL_DIR)" \
			  -DCMAKE_BUILD_TYPE=Release \
			  "$(LOCO_DIR)/$(PKG)" && \
		$(MAKE) -j$(nproc) && $(MAKE) install


# Clean only g1_locomotion-related build files
.PHONY: clean
clean:
	@echo " Cleaning g1_locomotion-related build and install files..."
	@if [ -d "$(BUILD_DIR)" ]; then \
		for pkg in $(LOCO_PACKAGES); do \
			rm -rf "$(BUILD_DIR)/$$pkg"; \
		done; \
	fi
	@if [ -d "$(INSTALL_DIR)" ]; then \
		for pkg in $(LOCO_PACKAGES); do \
			rm -rf "$(INSTALL_DIR)/share/$$pkg"; \
		done; \
	fi
	@echo "Clean completed."


# Force a rebuild of all g1_locomotion packages
.PHONY: rebuild
rebuild: clean build

# Source the workspace
.PHONY: source
source:
	@echo "source $(INSTALL_DIR)/setup.bash"

# Show available commands
.PHONY: help
help:
	@echo "make all          - Build all g1_locomotion packages"
	@echo "make build        - Same as 'make all' (builds all g1_locomotion packages)"
	@echo "make clean        - Remove only g1_locomotion-related build files"
	@echo "make rebuild      - Clean and rebuild all g1_locomotion packages"
	@echo "make pkg PKG=<name> - Build a specific package inside g1_locomotion"
	@echo "make source       - Print command to source the workspace"

