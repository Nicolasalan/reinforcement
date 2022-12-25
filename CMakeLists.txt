cmake_minimum_required(VERSION 2.8.3)
project(motion)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()
catkin_python_setup()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
