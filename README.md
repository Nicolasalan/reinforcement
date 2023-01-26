# Twin Delayed DDPG for environment navigation

This repository contains training neural networks for learning path planning using reinforcement learning, specifically # [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#id1) (DDPG) and its variance # [Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html#id1) (TD3).

The inputs to the model are data from the lidar sensors, distance to the target, theta angle and speed and have the speed of the robot as output. The entire environment was carried out in ROS Noetic and Gazebo 11.