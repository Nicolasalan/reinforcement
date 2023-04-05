#!/bin/bash
# Source ROS and Catkin workspaces
source /opt/ros/noetic/setup.bash
source /usr/share/gazebo-11/setup.sh

# Execute the command passed into this entrypoint
exec "$@"