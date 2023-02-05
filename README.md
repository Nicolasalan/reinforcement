# Twin Delayed DDPG for environment navigation

[![ROS Noetic Compatible](https://img.shields.io/badge/ROS-Noetic-yellow)](http://wiki.ros.org/noetic)
[![CI Build Status](https://github.com/Nicolasalan/motion/actions/workflows/main.yml/badge.svg?branch=main&event=status)](https://github.com/Nicolasalan/motion/actions/workflows/main.yml)
[![Docker](https://img.shields.io/badge/Docker-v20.10.21-blue)](https://docs.docker.com/)
[![Python3](https://img.shields.io/badge/Python-v3.8.10-brightgreen)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-v1.13.1-orange)](https://pytorch.org/)
[![Numpy](https://img.shields.io/badge/NumPy-v1.17.4-blueviolet)](https://numpy.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-v20.04-9cf)](https://releases.ubuntu.com/)
[![Cuda](https://img.shields.io/badge/Cuda-v11.8-red)](https://developer.nvidia.com/cuda-downloads)

This repository contains training neural networks for learning path planning using reinforcement learning, specifically [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#id1) (DDPG) and its variance [Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html#id1) (TD3).

The inputs to the model are data from the lidar sensors, distance to the target, theta angle and speed and have the speed of the robot as output. The entire environment was carried out in ROS Noetic and Gazebo v11.
