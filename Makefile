DOCKER_ENV_VARS = \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" 

DOCKER_VOLUMES = \
	--volume="$(shell pwd)":"/ws/src/motion":rw \
	--volume="$(shell pwd)/src/motion/checkpoints:/ws/src/motion/src/motion/checkpoints":rw \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="${HOME}/.Xauthority:/root/.Xauthority:rw" 
	
DOCKER_ARGS = ${DOCKER_VOLUMES} ${DOCKER_ENV_VARS}

define xhost_activate
	@echo "Enabling local xhost sharing:"
	@echo "  Display: ${DISPLAY}"
	@xhost local:root
endef

.PHONY: help
help:
	@echo '  help						--Display this help message'
	@echo '  update 					--Update ROS packages from git'
	@echo '  build 					--Build docker image for machine architecture'
	@echo '  clean 					--Docker image cleanup'
	@echo '  start						--Start training session'
	@echo '  terminal					--Start terminal in docker'
	@echo '  setup						--setup world and robot'
	@echo '  view						--setup view gazebo'
	@echo '  library					--Test library functions'
	@echo '  ros						--Test ROS topics'
	@echo '  sim						--Test Simulation Gazebo'
	@echo '  package					--Test Dependencies'
	@echo '  integration					--Test All'
	@echo '  tensorboard					--Start Tensorboard in localhost:6006'

# === Build docker ===
.PHONY: build
build:
	@echo "Building docker image ..."
	@docker login && docker build -t motion-docker . 
	@mkdir -p ${PWD}/src/motion/checkpoints
	@mkdir -p ${PWD}/src/motion/run

# === Clean docker ===
.PHONY: clean
clean:
	@echo "Closing all running docker containers ..."
	@sudo docker system prune -f

# === Run terminal docker ===
.PHONY: terminal
terminal:
	@echo "Terminal docker ..."
	@sudo xhost + 
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash

# === setup model ===
.PHONY: setup
setup:
	@echo "Setup world ..."
	@docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roslaunch motion bringup.launch"

# === setup view ===
.PHONY: view 
view:
	@echo "Setup View world ..."
	@sudo xhost +
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roslaunch motion view.launch"

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roslaunch motion start.launch"

# === Start train docker GPU ===
.PHONY: start-gpu
start-gpu:
	@echo "Starting training ..."
	@sudo docker run -it --net=host --gpus all ${DOCKER_ARGS} --env="NVIDIA_DRIVER_CAPABILITIES=all" motion-docker bash -c "source devel/setup.bash && roslaunch motion start.launch"

# === Test Library ===
.PHONY: library
library:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/library.py"

# === Test ROS ===
.PHONY: ros
ros:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/ros.py"

# === Test Simulation ===
.PHONY: sim
sim:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/sim.py"

# === Test Learning ===
.PHONY: package
package:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/package.py"

# === Test Full ===
.PHONY: integration
integration:
	@echo "Testing ..."
	@docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/ros.py && python3 test/library.py && python3 test/package.py && python3 test/sim.py"

# === Test workflow ===
.PHONY: workflow
workflow:
	@echo "Testing ..."
	@sudo docker login -u ninim && sudo docker run --net=host --volume=${PWD}:/ws/src/motion:rw ninim/motion-docker:latest bash -c "cd /ws/src/motion && ./workflow.sh"

# === Tensorboard ===
.PHONY: tensorboard
tensorboard:
	@echo "tensorboard ..."
	@sudo docker run -it --net=host -p 6006:6006 ${DOCKER_ARGS} motion-docker bash -c "cd /ws/src/motion/src/motion && tensorboard --logdir=run/"

# === Install Weights ===
.PHONY: install
install:
	@echo "Install Weights ..."
	@cd ${PWD}/src/motion/checkpoints && wget https://nicolasalan.github.io/data/checkpoints/critic_model.pth && wget https://nicolasalan.github.io/data/checkpoints/actor_model.pth