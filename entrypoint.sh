#!/bin/bash
# Source ROS and Catkin workspaces
source /opt/ros/noetic/setup.bash
source /usr/share/gazebo-11/setup.sh

# Set environment variables
export GAZEBO_MODEL_PATH=/ws/src/bookstore/models

# Execute the command passed into this entrypoint
exec "$@"