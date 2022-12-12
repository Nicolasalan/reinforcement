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

# === Start train docker ===
.PHONY: start 
start:
	@echo "Starting training"
	@sudo docker run -it --net=host motion-rl bash -c "source devel/setup.bash && roslaunch motion start.launch"
