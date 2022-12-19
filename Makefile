# setup
DOCKER_ENV_VARS= \
	--volume="$(PWD):/ws/src/motion":rw

# === Build docker ===
.PHONY: build
build:
	@echo "Building docker image ..."
	@docker login && docker build -t motion-docker  . 

# === Clean docker ===
.PHONY: clean
clean:
	@echo "Closing all running docker containers ..."
	@sudo docker system prune -f

# === Run terminal docker ===
.PHONY: terminal
terminal:
	@echo "Terminal docker ..."
	@sudo docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash

# === Spawn model ===
.PHONY: spawn 
spawn:
	@echo "Spawn model ..."
	@sudo docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roslaunch motion spawn.launch"

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training ..."
	@sudo docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source devel/setup.bash && roslaunch motion start.launch"

# === Start Rosboard docker ===
.PHONY: rosboard
rosboard:
	@echo "Starting rosboard ..."
	@sudo docker run -it --net=host ${DOCKER_ENV_VARS} motion-docker bash -c "source /opt/ros/noetic/setup.bash && ./ws/src/rosboard/run"