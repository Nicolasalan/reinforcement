# Twin Delayed DDPG for environment navigation

<p align="center">
  <a href="http://wiki.ros.org/noetic">
    <img src="https://img.shields.io/badge/ROS-Noetic-yellow" alt="ROS Noetic Compatible">
  </a>
  <a href="https://docs.docker.com/">
    <img src="https://img.shields.io/badge/Docker-v20.10.21-blue" alt="Docker">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-v3.8.10-brightgreen" alt="Python3">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-v1.13.1-orange" alt="Pytorch">
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-v1.17.4-blueviolet" alt="Numpy">
  </a>
  <a href="https://releases.ubuntu.com/">
    <img src="https://img.shields.io/badge/Ubuntu-v20.04-9cf" alt="Ubuntu">
  </a>
  <a href="https://developer.nvidia.com/cuda-downloads">
    <img src="https://img.shields.io/badge/Cuda-v11.8-red" alt="Cuda">
  </a>
</p>


This repository contains training neural networks for learning path planning using reinforcement learning, specifically [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#id1) (DDPG) and its variance [Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html#id1) (TD3).

The inputs to the model are data from the lidar sensors, distance to the target, theta angle and speed and have the speed of the robot as output. The entire environment was carried out in ROS Noetic and Gazebo v11.
