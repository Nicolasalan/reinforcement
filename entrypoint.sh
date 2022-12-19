#!/bin/bash
# Basic entrypoint for ROS / Catkin Docker containers

# Source ROS and Catkin workspaces
source /opt/ros/noetic/setup.bash

echo "source /ws/devel/setup.bash" >> ~/.bashrc

echo "Sourced Catkin workspace!"

# Set environment variables
echo "Export world model path"
export GAZEBO_MODEL_PATH=/ws/src/aws-robomaker-bookstore-world/models

# Execute the command passed into this entrypoint
exec "$@"