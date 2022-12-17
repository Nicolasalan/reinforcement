#!/usr/bin/env python3

# importar bibliotecas comuns
import os
import rospy
import numpy as np
import random
import yaml
import math
import time

# importar mensagens do ROS
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion

# importar utilitarios
from utils import angles, distance_to_goal, random_goal, path_goal

# folder to load config file
CONFIG_PATH = "/ws/src/motion/config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        param = yaml.safe_load(file)

    return param

param = load_config("main_config.yaml")

class Env():
     def __init__(self):

          # definir o estado inicial
          self.collision_dist = param["collision_dist"]
          self.goal_model = param["goal_model"]
          self.goal_reached_dist = param["goal_reached_dist"]
          self.environment_dim = param["environment_dim"]
          self.time_delta = param["time_delta"]

          # inicializar variaveis globais
          self.odom_x = 0.0
          self.odom_y = 0.0
          self.goal_x = 0.0
          self.goal_y = 0.0

          self.scan_data = np.ones(self.environment_dim) * 10
          self.goals = path_goal()
          self.last_odom = None

          ##### publicacoes e assinaturas do ROS #####
          self.pub_cmd_vel = rospy.Publisher(param["topic_cmd"], Twist, queue_size=10)
          self.odom = rospy.Subscriber(param["topic_odom"], Odometry, self.odom_callback, queue_size=1)
          self.scan = rospy.Subscriber(param["scan"], LaserScan, self.scan_callback)

          ##### servicos do ROS #####
          self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
          self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
          self.state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

     def odom_callback(self, od_data):
          self.last_odom = od_data

     def scan_callback(self, scan):
          data = [(point[0], point[1]) for point in scan.ranges]
          self.scan_data = np.ones(self.environment_dim) * 10
          for i in range(len(data)):
               dot = data[i][0] * 1 + data[i][1] * 0
               mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
               mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
               beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
               dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2)

               for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                         self.scan_data[j] = min(self.scan_data[j], dist)
                         break

     # Perform an action and read a new state
     def step(self, action):
          target = False

          # Publish the robot action
          vel_cmd = Twist()
          vel_cmd.linear.x = action[0]
          vel_cmd.angular.z = action[1]
          self.vel_pub.publish(vel_cmd)

          rospy.wait_for_service("/gazebo/unpause_physics")
          try:
               self.unpause()
          except (rospy.ServiceException) as e:
               print("/gazebo/unpause_physics service call failed")

          # propagate state for TIME_DELTA seconds
          time.sleep(self.time_delta)

          rospy.wait_for_service("/gazebo/pause_physics")
          try:
               pass
               self.pause()
          except (rospy.ServiceException) as e:
               print("/gazebo/pause_physics service call failed")

          # read velodyne laser state
          done, collision, min_laser = self.observe_collision(self.scan_data)
          v_state = []
          v_state[:] = self.scan_data[:]
          laser_state = [v_state]

          # Calculate robot heading from odometry data
          self.odom_x = self.last_odom.pose.pose.position.x
          self.odom_y = self.last_odom.pose.pose.position.y
          quaternion = Quaternion(
               self.last_odom.pose.pose.orientation.w,
               self.last_odom.pose.pose.orientation.x,
               self.last_odom.pose.pose.orientation.y,
               self.last_odom.pose.pose.orientation.z,
          )
          euler = quaternion.to_euler(degrees=False)
          angle = round(euler[2], 4)

          # Calculate distance to the goal from the robot
          distance = distance_to_goal(self.odom_x, self.goal_x, self.odom_y, self.goal_y)

          # Calculate the relative angle between the robots heading and heading toward the goal
          theta = angles(self.odom_x, self.goal_x, self.odom_y, self.goal_y, angle)

          # Detect if the goal has been reached and give a large positive reward
          if distance < self.goal_reached_dist:
               target = True
               done = True

          robot_state = [distance, theta, action[0], action[1]]
          state = np.append(laser_state, robot_state)
          reward = self.get_reward(target, collision, action, min_laser)
          return state, reward, done, target

     def reset(self):
     
          # Resets the state of the environment and returns an initial observation.
          rospy.wait_for_service("/gazebo/reset_simulation")
          try:
               self.reset()

          except rospy.ServiceException as e:
               print("/gazebo/reset_simulation service call failed")

          # set a random goal in empty space in environment
          goal = self.goals
          self.goal_x, self.goal_y = random_goal(goal)

          box_state = ModelState()
          box_state.model_name = "target"
          box_state.pose.position.x = self.goal_x
          box_state.pose.position.y = self.goal_y
          box_state.pose.orientation.x = 0.0
          box_state.pose.orientation.y = 0.0
          box_state.pose.orientation.z = 0.0
          box_state.pose.orientation.w = 1.0
          self.state.publish(box_state)
          print("Target randomized")
               
          rospy.wait_for_service("/gazebo/unpause_physics")
          try:
               self.unpause()
          except (rospy.ServiceException) as e:
               print("/gazebo/unpause_physics service call failed")

          time.sleep(self.time_delta)

          rospy.wait_for_service("/gazebo/pause_physics")
          try:
               self.pause()
          except (rospy.ServiceException) as e:
               print("/gazebo/pause_physics service call failed")

          
          v_state = []
          v_state[:] = self.scan_data[:]
          laser_state = [v_state]

          # Calculate distance to the goal from the robot
          distance = distance_to_goal(self.odom_x, self.goal_x, self.odom_y, self.goal_y)

          # Calculate the relative angle between the robots heading and heading toward the goal
          theta = angles(self.odom_x, self.goal_x, self.odom_y, self.goal_y, angle)

          robot_state = [distance, theta, 0.0, 0.0]
          state = np.append(laser_state, robot_state)
          return state

     @staticmethod
     def get_reward(self, target, collision, action, min_laser):
          if target:
               return 100.0
          elif collision:
               return -100.0
          else:
               r3 = lambda x: 1 - x if x < 1 else 0.0
               return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

     @staticmethod
     def observe_collision(self, laser_data):
          # Detect a collision from laser data
          min_laser = min(laser_data)
          if min_laser < self.collision_dist:
               return True, True, min_laser
          return False, False, min_laser

     @staticmethod
     def change_goal(self):
          # Place a new goal and check if its location is not on one of the obstacles
          self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
          self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
     