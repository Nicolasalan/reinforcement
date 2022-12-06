# === Build raspberry docker ===
.PHONY: docker-build
docker-build:
	@echo "Building docker image"
	@sudo docker build -t drl-docker . 

# === Clean docker ===
.PHONY: docker-clean
docker-clean:
	@echo "Closing all running docker containers"
	@sudo docker system prune -f

# === Run terminal docker ===
.PHONY: run-docker
run-docker:
	@echo "Running docker container"
	@sudo docker run -it --net=multihost -p 8080:8080 drl-docker bash

# === Delete port ===
.PHONY: delete-port
delete-port:
	@echo "Deleting port"
	@sudo fuser -k 8080/tcp
	@docker stop $(docker ps -a -q)

# === Create network ===
.PHONY: create-network
create-network:
	@echo "Creating network"
	@sudo docker swarm init
	@sudo docker network create -d overlay --attachable multihost