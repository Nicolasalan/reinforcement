#!/bin/bash
sudo apt-get update
sudo apt-get install -y ros-noetic-ros-controllers 
sudo apt-get install -y ros-noetic-joint-state-controller
sudo apt-get install -y ros-noetic-joint-state-publisher
sudo apt-get install -y ros-noetic-robot-state-publisher
sudo apt-get install -y ros-noetic-robot-state-controller
sudo apt-get install -y ros-noetic-xacro 
sudo apt-get install -y ros-noetic-smach-ros
sudo apt-get install -y ros-noetic-gazebo-ros
sudo apt-get install -y ros-noetic-gazebo-ros-control
sudo apt-get install -y ros-noetic-rplidar-ros
sudo apt-get install -y ros-noetic-driver-base
sudo apt-get install -y ros-noetic-rosserial-arduino
sudo apt-get install -y ros-noetic-map-server
sudo apt-get install -y ros-noetic-gazebo-ros-pkgs

pip3 install torch

sudo apt-get install -q -y --no-install-recommends
sudo apt-get install -q -y --no-install-recommends build-essential
sudo apt-get install -q -y --no-install-recommends apt-utils
sudo apt-get install -q -y --no-install-recommends cmake
sudo apt-get install -q -y --no-install-recommends g++
sudo apt-get install -q -y --no-install-recommends git
sudo apt-get install -q -y --no-install-recommends libcanberra-gtk
sudo apt-get install -q -y --no-install-recommends python3-catkin-tools
sudo apt-get install -q -y --no-install-recommends python3-pip
sudo apt-get install -q -y --no-install-recommends python3-tk
sudo apt-get install -q -y --no-install-recommends python3-yaml
sudo apt-get install -q -y --no-install-recommends python3-dev
sudo apt-get install -q -y --no-install-recommends python3-numpy
sudo apt-get install -q -y --no-install-recommends python3-rosinstall
sudo apt-get install -q -y --no-install-recommends python3-catkin-pkg
sudo apt-get install -q -y --no-install-recommends python3-rosdistro
sudo apt-get install -q -y --no-install-recommends python3-rospkg
sudo apt-get install -q -y --no-install-recommends wget
sudo apt-get install -q -y --no-install-recommends curl
sudo apt-get install -q -y --no-install-recommends vim 