FROM osrf/ros:noetic-desktop-full

# Change the default shell to Bash
SHELL [ "/bin/bash" , "-c" ]

# Setup minimal
ARG DEBIAN_FRONTEND=noninteractive 

# Install basic apt packages
RUN apt-get update && apt-get install -y \
  cmake \
  apt-utils \
  g++ \
  gnupg gnupg1 gnupg2 \
  libcanberra-gtk* \
  python3-catkin-tools \
  python3-pip \
  python3-tk \
  python3-yaml \
  python3-dev \
  python3-numpy \
  wget \
  git \
  curl \
  xterm \
  npm

# depenpencies gzweb
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    build-essential \
    imagemagick \
    libboost-all-dev \
    libgts-dev \
    libjansson-dev \
    libtinyxml-dev \
    mercurial \
    nodejs \
    pkg-config \
    psmisc \
    xvfb

# Install dependencies
RUN apt-get update && apt-get install -y ros-noetic-ros-controllers \
 && apt-get install -y ros-noetic-joint-state-controller \
 && apt-get install -y ros-noetic-joint-state-publisher \
 && apt-get install -y ros-noetic-robot-state-publisher \
 && apt-get install -y ros-noetic-robot-state-controller \
 && apt-get install -y ros-noetic-xacro \ 
 && apt-get install -y ros-noetic-smach-ros \
 && apt-get install -y ros-noetic-teleop-twist-keyboard \
 && apt-get install -y ros-noetic-gazebo-ros \
 && apt-get install -y ros-noetic-gazebo-ros-control \
 && apt-get install -y ros-noetic-rplidar-ros \
 && apt-get install -y ros-noetic-driver-base \
 && apt-get install -y ros-noetic-rosserial-arduino

# create a python3.9 symlink
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install torch latest
RUN python3 -m pip --no-cache-dir install \
    torch 

# Gzweb 
RUN apt-get clean

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys D2486D2DD83DB69272AFE98867170598AF249743

# setup sources.list
RUN . /etc/os-release \
    && echo "deb http://packages.osrfoundation.org/gazebo/$ID-stable `lsb_release -sc` main" > /etc/apt/sources.list.d/gazebo-latest.list

# install gazebo
RUN apt-get install -y libgazebo11 gazebo11

#install gazebo packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgazebo11-dev=11.12.0-1* 

# clone gzweb
ENV GZWEB_WS /root/gzweb
RUN git clone -b master https://github.com/osrf/gzweb $GZWEB_WS

# setup environment
EXPOSE 8080
EXPOSE 7681

# create a catkin workspace
RUN mkdir -p /ws/src \
 && cd /ws/src \
 && source /opt/ros/noetic/setup.bash \
 && catkin_init_workspace 

# Copy the source files
WORKDIR /ws
VOLUME . /ws/src/motion-rl

# Rosdep
RUN apt-get install python3-rosdep \
 && rm /etc/ros/rosdep/sources.list.d/20-default.list \
 && sudo rosdep init \
 && rosdep update 

# Build the Catkin workspace
RUN cd /ws \
 && source /opt/ros/noetic/setup.bash \
 && rosdep install -y --from-paths src --ignore-src \
 && catkin build

# Install Gzweb
RUN cd /root/gzweb && source /usr/share/gazebo-11/setup.sh && npm run deploy

# Setup bashrc
RUN echo "source /ws/devel/setup.bash" >> ~/.bashrc \
 && echo "source /usr/share/gazebo-11/setup.bash" >> ~/.bashrc 

# Remove display warnings
RUN mkdir /tmp/runtime-root
ENV XDG_RUNTIME_DIR "/tmp/runtime-root"
ENV NO_AT_BRIDGE 1

# Install python dependencies
#RUN cd /src/motion-rl && pip3 install -r requirements.txt

# entrypoint script
#ENTRYPOINT [ "/src/motion-rl/entrypoint.sh" ]