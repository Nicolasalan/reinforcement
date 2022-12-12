# === Build docker ===
.PHONY: build
build:
	@echo "Building docker image"
	@sudo docker build -t motion-rl  . 

# === Clean docker ===
.PHONY: clean
clean:
	@echo "Closing all running docker containers"
	@sudo docker system prune -f

# === Run terminal docker ===
.PHONY: terminal
terminal:
	@echo "Terminal docker"
	@sudo docker run -it --net=host motion-rl bash

# === Spawn model docker ===
.PHONY: spawn 
spawn:
	@echo "Spawn model"
	@sudo docker run -it --net=host motion-rl bash -c "source devel/setup.bash && roslaunch motion_rl spawn.launch"

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training"
	@sudo docker run -it --net=host motion-rl bash -c "source devel/setup.bash && roslaunch motion_rl start.launch"

# === Delete port ===
.PHONY: delete
delete:
	@echo "Deleting port"
	@sudo fuser -k 8080/tcp
	@docker stop $(docker ps -a -q)

# === Create network ===
.PHONY: network
network:
	@echo "Creating network"
	@sudo docker swarm init
	@sudo docker network create -d overlay --attachable multihost