DOCKER_ENV_VARS = \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" 

DOCKER_VOLUMES = \
	--volume="$(shell pwd)":"/ws/src/motion":rw \
	--volume="$(shell pwd)/src/motion/checkpoints:/ws/src/motion/src/motion/checkpoints":rw \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="${HOME}/.Xauthority:/root/.Xauthority:rw" 
	
DOCKER_ARGS = ${DOCKER_VOLUMES} ${DOCKER_ENV_VARS}

COMMAND="source devel/setup.bash && roslaunch motion bringup.launch"
COMMAND_VIEW="source devel/setup.bash && roslaunch motion view.launch"

define xhost_activate
	@echo "Enabling local xhost sharing:"
	@echo "  Display: ${DISPLAY}"
	@xhost local:root
endef

# === Build docker ===
.PHONY: build
build:
	@echo "Building docker image ..."
	@sudo docker build -t motion-docker  . 

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
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c ${COMMAND}

# === setup view ===
.PHONY: view 
view:
	@echo "Setup View world ..."
	@sudo xhost +
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c ${COMMAND_VIEW}

# === Spawn model ===
.PHONY: spawn 
spawn:
	@echo "Spawn Model ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roslaunch motion spawn.launch"

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roslaunch motion start.launch"

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
	@sudo docker run -it --net=host ${DOCKER_ARGS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/ros.py && python3 test/library.py && python3 test/package.py && python3 test/sim.py"

# === Start Rosboard docker ===
.PHONY: rosboard
rosboard:
	@echo "Starting rosboard ..."
	@echo "Access http://localhost:8888"
	@sudo docker run -it --net=host -p 8888:8888 ${DOCKER_ARGS} motion-docker bash -c "source /opt/ros/noetic/setup.bash && cd src/rosboard && ./run"
