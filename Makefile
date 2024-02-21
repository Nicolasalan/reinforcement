DOCKER = True
DOCKER_ENV_VARS = \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" 

# --volume="$(shell pwd)":"/ws/src/reinforcement":rw \
# --volume="$(shell pwd)/src/reinforcement/checkpoints:/ws/src/reinforcement/src/reinforcement/checkpoints":rw \

DOCKER_VOLUMES = \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="${HOME}/.Xauthority:/root/.Xauthority:rw" 

DOCKER_GPU = \
	--gpus all \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
	--env="NVIDIA_VISIBLE_DEVICES all" 
	
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
	@echo '  install					--Install Weights'
	@echo '  start-all					--Start Simulation and Training'
	@echo '  server					--Start Training Server'
	@echo '  start-gpu					--Start Training GPU'
	@echo '  waypoint					--Setup Waypoint'

#########################################################################################################################
################################################ INSTALL ################################################################
#########################################################################################################################

# === Build docker ===
.PHONY: build
build:
	@echo "Building docker image ..."
	@docker build -t reinforcement-docker . 
	
# @sudo mkdir -p ${PWD}/src/reinforcement/checkpoints
# @sudo mkdir -p ${PWD}/src/reinforcement/run
# @sudo mkdir -p ${PWD}/config/map

.PHONY: npm
npm:
	@echo "Installing npm ..."
	@sudo install npm
	@sudo npm install -g n
	@sudo n stable
	@sudo npm install -g npm@latest
	@sudo npm install -g vitepress

# === Install Weights ===
.PHONY: weights
weights:
	@echo "Install Weights ..."
	@cd ${PWD}/src/reinforcement/checkpoints && wget https://nicolasalan.github.io/data/checkpoints/critic_model.pth && wget https://nicolasalan.github.io/data/checkpoints/actor_model.pth

# === Setup Waypoint ===
.PHONY: waypoint
waypoint:
	@echo "Setup Waypoint and Create Env..."
	@sudo xhost + 
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roslaunch reinforcement setup.launch map_file:=/ws/src/reinforcement/config/map/map.yaml"

########################################################################################################################
################################################ USAGE #################################################################
########################################################################################################################

# === Clean docker ===
.PHONY: clean
clean:
	@echo "Closing all running docker containers ..."
	@sudo docker system prune -f

# === setup model ===
.PHONY: setup
setup:
	@echo "Setup world ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roslaunch reinforcement bringup.launch"

# === Run terminal docker ===
.PHONY: terminal
terminal:
	@echo "Terminal docker ..."
	@sudo xhost + 
	@docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash

# === setup view ===
.PHONY: view 
view:
	@echo "Setup View World ..."
	@sudo xhost +
	@sudo docker run -it --net=host ${DOCKER_ARGS} --memory=40g reinforcement-docker bash -c "source devel/setup.bash && roslaunch reinforcement view.launch"

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} --memory=40g reinforcement-docker bash -c "source devel/setup.bash && roslaunch reinforcement start.launch"

# === Start All Training ===
.PHONY: start-all
start-all:
	@echo "Starting training All ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "cd /ws && source devel/setup.bash && roslaunch reinforcement bringup.launch & sleep 20 && cd /ws && source devel/setup.bash && roslaunch reinforcement start.launch"

# === Start Training GPU ===
.PHONY: start-gpu
start-gpu:
	@echo "Starting training in GPU ..."
	@sudo docker run -it --net=host --gpus all ${DOCKER_GPU} ${DOCKER_ARGS} reinforcement-docker bash -c "cd /ws && source devel/setup.bash && roslaunch reinforcement bringup.launch & sleep 20 && cd /ws && source devel/setup.bash && roslaunch reinforcement start.launch"

# === Tensorboard ===
.PHONY: board
board:
	@echo "tensorboard ..."
	@sudo docker run -it --net=host -p 6006:6006 ${DOCKER_ARGS} reinforcement-docker bash -c "tensorboard --logdir=/ws/src/reinforcement/src/reinforcement/logs"

########################################################################################################################
################################################ TESTS #################################################################
########################################################################################################################

# === Test functions ===
.PHONY: func
func:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roscd reinforcement && python3 test/func.py"

# === Test ROS ===
.PHONY: ros
ros:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roscd reinforcement && python3 test/ros.py"

# === Test Simulation ===
.PHONY: sim
sim:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roscd reinforcement && python3 test/sim.py"

# === Test Package ===
.PHONY: package
package:
	@echo "Testing ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roscd reinforcement && python3 test/package.py"

# === Test Full ===
.PHONY: integration
integration:
	@echo "Testing ..."
	@docker run -it --net=host ${DOCKER_ARGS} reinforcement-docker bash -c "source devel/setup.bash && roscd reinforcement && python3 test/ros.py && python3 test/functions.py && python3 test/package.py && python3 test/sim.py"