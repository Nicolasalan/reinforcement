LOCAL: False
VISUALIZE: False

# setup
DOCKER_ENV_VARS= \
	--volume="$(shell pwd):/ws/src/motion":rw \
	--volume="$(shell pwd)/src/motion/checkpoints:/ws/src/motion/src/motion/checkpoints":rw \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix" \
	--volume="${HOME}/.Xauthority:/root/.Xauthority:rw" \
	--env="DISPLAY=${DISPLAY}" \
	--env="QT_X11_NO_MITSHM=1" \
	--ipc=host \
	--privileged 	

COMMAND="source devel/setup.bash && roslaunch motion bringup.launch"

define xhost_activate
	@echo "Enabling local xhost sharing:"
	@echo "  Display: ${DISPLAY}"
	@xhost local:root
endef

# === Build docker ===
.PHONY: build
build:
	@echo "Building docker image ..."
	@docker login && docker build -t motion-docker  . 

# === Clean docker ===
.PHONY: clean
clean:
	@echo "Closing all running docker containers ..."
	@docker system prune -f

# === Run terminal docker ===
.PHONY: terminal
terminal:
	@echo "Terminal docker ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash

# === setup model ===
.PHONY: setup 
setup:
	@echo "Setup world ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c ${COMMAND}

# === Spawn model ===
.PHONY: spawn 
spawn:
	@echo "Spawn Model ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roslaunch motion spawn.launch"

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roslaunch motion start.launch"

# === Test Library ===
.PHONY: test-library
test-library:
	@echo "Testing ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/library.py"

# === Test ROS ===
.PHONY: test-ros
test-ros:
	@echo "Testing ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/ros.py"

# === Test Learning ===
.PHONY: test-package
test-package:
	@echo "Testing ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/package.py"

# === Test Full ===
.PHONY: test-full
test-full:
	@echo "Testing ..."
	@docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roscd motion && python3 test/library.py && python3 test/ros.py"

# === Start Rosboard docker ===
.PHONY: rosboard
rosboard:
	@echo "Starting rosboard ..."
	@echo "Access http://localhost:8888"
	@docker run -it --net=host -p 8888:8888 ${DOCKER_ENV_VARS} motion-docker bash -c "source /opt/ros/noetic/setup.bash && cd src/rosboard && ./run"
