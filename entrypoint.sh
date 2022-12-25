#!/bin/bash
# Source ROS and Catkin workspaces
source /opt/ros/noetic/setup.bash
source /usr/share/gazebo-11/setup.sh

echo "Sourced Catkin workspace!"

# Set environment variables
export GAZEBO_MODEL_PATH=/ws/src/aws-robomaker-bookstore-world/models
echo "Export world model path"

# Execute the command passed into this entrypoint
exec "$@"