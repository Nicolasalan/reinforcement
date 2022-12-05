FROM osrf/ros:noetic-desktop-full

# Change the default shell to Bash
SHELL [ "/bin/bash" , "-c" ]

# Setup minimal
RUN apt-get update \
   && apt-get install -y --no-install-recommends apt-utils
ARG DEBIAN_FRONTEND=noninteractive 

# Install basic apt packages
RUN apt-get update && apt-get install -y \
  cmake \
  g++ \
  gnupg gnupg1 gnupg2 \
  libcanberra-gtk* \
  python3-catkin-tools \m
  python3-pip \
  python3-tk \
  wget \
  npm \
  vim \
  curl

# Install Git
RUN apt-get update && apt-get install -y git && apt-get install -y build-essential

# Install dependencies
RUN apt-get update && apt-get install -y ros-noetic-ros-controllers \
 && apt-get install -y ros-noetic-joint-state-controller \
 && apt-get install -y ros-noetic-joint-state-publisher \
 && apt-get install -y ros-noetic-xacro \ 
 && apt-get install -y ros-noetic-smach-ros \
 && apt-get install -y ros-noetic-teleop-twist-keyboard \
 && apt-get install -y ros-noetic-gazebo-ros \
 && apt-get install -y ros-noetic-gazebo-ros-control \
 && apt-get install -y ros-noetic-rplidar-ros

RUN source /opt/ros/noetic/setup.bash \
 && mkdir -p /ws/src 

COPY . /ws/src

RUN cd /ws \
 && source /opt/ros/noetic/setup.bash \
 && rosdep install -y --from-paths src --ignore-src \
 && catkin build

RUN echo "source /ws/devel/setup.bash" >> ~/.bashrc \
 && echo "source /usr/share/gazebo/setup.bash" >> ~/.bashrc 

WORKDIR /ws

# Remove display warnings
RUN mkdir /tmp/runtime-root
ENV XDG_RUNTIME_DIR "/tmp/runtime-root"
ENV NO_AT_BRIDGE 1