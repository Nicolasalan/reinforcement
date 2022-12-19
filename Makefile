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
	@echo "Access http://localhost:8888"
	@sudo docker run -it ${DOCKER_ENV_VARS} motion-docker bash -c "source /opt/ros/noetic/setup.bash && cd src/rosboard && ./run"

# === Delete port ===
.PHONY: delete-port
delete-port:
	@echo "Deleting port"
	@sudo fuser -k 8080/tcp
	@docker stop $(docker ps -a -q)

# === Create network ===
.PHONY: network
network:
	@echo "Creating network"
	@sudo docker swarm init
	@sudo docker network create -d overlay --attachable multihost