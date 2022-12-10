#!/bin/bash

# Source ROS and Catkin workspaces
source /opt/ros/noetic/setup.bash
echo "source /ws/devel/setup.bash" >> ~/.bashrc
source /ws/devel/setup.bash

echo "Sourced Catkin workspace!"

roslaunch motion start.launch