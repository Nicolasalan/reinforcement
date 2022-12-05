#!/bin/bash

# Source ROS and Catkin workspaces
source /opt/ros/noetic/setup.bash
echo "source /ws/devel/setup.bash" >> ~/.bashrc
source /ws/devel/setup.bash

echo "Sourced Catkin workspace!"

# Set environment variables
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(rospack find motion-rl)/models

roslaunch motion-rl start.launch & \
cd ~/gzweb && npm start
# Execute the command passed into this entrypoint
exec "$@"