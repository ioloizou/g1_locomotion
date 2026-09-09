SHELL := /bin/bash

WS_DIR      := /home/forest_ws
LOCO_DIR    := $(WS_DIR)/src/g1_locomotion
BUILD_DIR   := $(WS_DIR)/build
INSTALL_DIR := $(WS_DIR)/install

# Build order matters: the message package generates headers the others include,
# and g1_mujoco_sim's launch check needs g1_mpc's node to exist.
PACKAGES := g1_description g1_msgs g1_mpc g1_mujoco_sim

SETUP := source /opt/ros/noetic/setup.bash && source $(WS_DIR)/setup.bash

.PHONY: all build clean rebuild test check-config help $(PACKAGES)

all: build

build: $(PACKAGES)

# Serialised on purpose; each package is still built with a parallel make.
$(PACKAGES):
	@echo "==> $@"
	@mkdir -p "$(BUILD_DIR)/$@"
	@cd "$(BUILD_DIR)/$@" && $(SETUP) && \
		cmake -DCMAKE_INSTALL_PREFIX="$(INSTALL_DIR)" \
		      -DCMAKE_BUILD_TYPE=Release \
		      "$(LOCO_DIR)/$@" > /dev/null && \
		$(MAKE) -j$$(nproc) install > /dev/null
	@echo "    ok"

g1_msgs: g1_description
g1_mpc: g1_msgs
g1_mujoco_sim: g1_mpc

clean:
	@for pkg in $(PACKAGES); do \
		rm -rf "$(BUILD_DIR)/$$pkg" "$(INSTALL_DIR)/share/$$pkg"; \
	done
	@echo "cleaned"

rebuild: clean build

# Controller checks that need neither a ROS master nor a display.
test: build
	@echo "==> MPC / gait / footstep self-test"
	@$(BUILD_DIR)/g1_mpc/devel/lib/g1_mpc/mpc_selftest
	@echo "==> PD stand smoke test"
	@$(SETUP) && python3 $(LOCO_DIR)/g1_mujoco_sim/scripts/pd_stand_demo.py \
		--headless --duration 3

# Fails if the configuration no longer matches the URDF.
check-config:
	@$(SETUP) && python3 $(LOCO_DIR)/g1_mujoco_sim/scripts/extract_robot_params.py --check

help:
	@echo "make build        - build and install every package, in dependency order"
	@echo "make clean        - remove this repository's build and install output"
	@echo "make rebuild      - clean, then build"
	@echo "make test         - run the offline controller checks"
	@echo "make check-config - verify config/g1_locomotion.yaml matches the URDF"
