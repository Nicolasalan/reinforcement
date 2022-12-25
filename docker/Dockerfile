# Ubuntu 20.04 image with NVIDIA CUDA + OpenGL and ROS Noetic
FROM osrf/ros:noetic-desktop-full

# Change the default shell to Bash
SHELL [ "/bin/bash" , "-c" ]

# Setup minimal
RUN apt-get update

# Install dependencies
RUN apt-get install -q -y --no-install-recommends \
  build-essential \
  apt-utils \
  cmake \
  g++ \
  git \
  libcanberra-gtk* \
  python3-catkin-tools \
  python3-pip \
  python3-tk \
  python3-yaml \
  python3-dev \
  python3-numpy \
  python3-rosinstall \
  python3-catkin-pkg \
  python3-rosdistro \
  python3-rospkg \
  wget \
  curl \
  vim

# Install dependencies ros
RUN apt-get update && apt-get install -y ros-noetic-ros-controllers \
 && apt-get install -y ros-noetic-joint-state-controller \
 && apt-get install -y ros-noetic-joint-state-publisher \
 && apt-get install -y ros-noetic-robot-state-publisher \
 && apt-get install -y ros-noetic-robot-state-controller \
 && apt-get install -y ros-noetic-xacro \ 
 && apt-get install -y ros-noetic-smach-ros \
 && apt-get install -y ros-noetic-gazebo-ros \
 && apt-get install -y ros-noetic-gazebo-ros-control \
 && apt-get install -y ros-noetic-rplidar-ros \
 && apt-get install -y ros-noetic-driver-base \
 && apt-get install -y ros-noetic-rosserial-arduino

# install pytorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install pyyaml matplotlib numpy squaternion

# create a catkin workspace
RUN git clone -b main https://github.com/Nicolasalan/DRL-robot-navigation.git

# Set the working directory
WORKDIR /DRL-robot-navigation

# Build the Catkin workspace
RUN cd catkin_ws \
 && source /opt/ros/noetic/setup.bash \
 && catkin_make_isolated

RUN export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch

# Setup bashrc
RUN cd catkin_ws && \
     echo "source /devel_isolated/setup.bash" >> ~/.bashrc 
