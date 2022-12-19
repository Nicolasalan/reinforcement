#!/usr/bin/env python3

# importar bibliotecas comuns
import os
import rospy
import numpy as np
import yaml
import math
import time

# importar mensagens do ROS
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState

from std_srvs.srv import Empty
from squaternion import Quaternion

# importar utilitarios
from utils import Extension

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
          self.goal_model = param["goal_model"]
          self.goal_reached_dist = param["goal_reached_dist"]
          self.environment_dim = param["environment_dim"]
          self.time_delta = param["time_delta"]
          self.util = Extension()

          # inicializar variaveis globais
          self.odom_x = 0.0
          self.odom_y = 0.0
          self.goal_x = 0.0
          self.goal_y = 0.0

          self.scan_data = np.ones(self.environment_dim) * 10
          self.goals = self.util.path_goal()
          self.last_odom = None

          ##### publicacoes e assinaturas do ROS #####
          self.pub_cmd_vel = rospy.Publisher(param["topic_cmd"], Twist, queue_size=10)
          self.odom = rospy.Subscriber(param["topic_odom"], Odometry, self.odom_callback, queue_size=1)
          self.scan = rospy.Subscriber(param["topic_scan"], LaserScan, self.scan_callback)

          ##### servicos do ROS #####
          self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
          self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

          self.gaps = [[-np.pi / 2, -np.pi / 2 + np.pi / self.environment_dim]]
          for m in range(self.environment_dim - 1):
               self.gaps.append(
                    [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
               )

          rospy.loginfo("Environment initialized")
          message = rospy.wait_for_message("/base_scan_front", LaserScan)
          rospy.loginfo(message)
          
     def odom_callback(self, od_data):
          self.last_odom = od_data

     def scan_callback(self, scan):
          data = scan.ranges
          self.scan_data = np.ones(self.environment_dim) * 10
          for point in data:
               dot = point * 1
               mag1 = math.sqrt(point ** 2)
               mag2 = 1
               beta = math.acos(dot / (mag1 * mag2)) * np.sign(0)
               dist = math.sqrt(point ** 2)

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
          self.pub_cmd_vel.publish(vel_cmd)

          rospy.wait_for_service("/gazebo/unpause_physics")
          try:
               self.unpause()

          except (rospy.ServiceException) as e:
               print("/gazebo/unpause_physics service call failed")

          # propagate state for TIME_DELTA seconds
          time.sleep(self.time_delta)

          rospy.wait_for_service("/gazebo/pause_physics")
          try:
               self.pause()

          except (rospy.ServiceException) as e:
               print("/gazebo/pause_physics service call failed")

          # read scan laser state
          done, collision, min_laser = self.util.observe_collision(self.scan_data)
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
          distance = self.util.distance_to_goal(self.odom_x, self.goal_x, self.odom_y, self.goal_y)

          # Calculate the relative angle between the robots heading and heading toward the goal
          theta = self.util.angles(self.odom_x, self.goal_x, self.odom_y, self.goal_y, angle)

          # Detect if the goal has been reached and give a large positive reward
          if distance < self.goal_reached_dist:
               target = True
               done = True

          robot_state = [distance, theta, action[0], action[1]]
          state = np.append(laser_state, robot_state)
          reward = self.util.get_reward(target, collision, action, min_laser)
          return state, reward, done, target

     def reset(self):
     
          # Resets the state of the environment and returns an initial observation.
          rospy.wait_for_service("/gazebo/reset_simulation")
          try:
               self.reset()

          except rospy.ServiceException as e:
               print("/gazebo/reset_simulation service call failed")

          angle = np.random.uniform(-np.pi, np.pi)
          quaternion = Quaternion.from_euler(0.0, 0.0, angle)
          rospy.logdebug("Set the robot pose")
          # set a random robot in empty space in environment
          path = self.goals
          x, y = 0.0, 0.0
          
          while True:
               x, y = self.util.random_goal(path)
               _x, _y = self.util.change_goal(path, self.odom_x, self.odom_y)
               check = self.util.check_pose(x, y, _x, _y)
               if check == True:
                    break
          
          rospy.logdebug("Position random goal")
          robot = ModelState()
          robot.model_name = param["robot"]
          robot.pose.position.x = x
          robot.pose.position.y = y
          robot.pose.orientation.x = quaternion.x
          robot.pose.orientation.y = quaternion.y
          robot.pose.orientation.z = quaternion.z
          robot.pose.orientation.w = quaternion.w
          self.set_state.publish(robot)

          self.goal_x, self.goal_y = _x, _y
               
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

          rospy.logdebug("Physics paused")
          
          v_state = []
          v_state[:] = self.scan_data[:]
          laser_state = [v_state]

          # Calculate distance to the goal from the robot
          distance = self.util.distance_to_goal(self.odom_x, self.goal_x, self.odom_y, self.goal_y)

          # Calculate the relative angle between the robots heading and heading toward the goal
          theta = self.util.angles(self.odom_x, self.goal_x, self.odom_y, self.goal_y, angle)

          rospy.logdebug("Calculate distance and angle")

          robot_state = [distance, theta, 0.0, 0.0]
          state = np.append(laser_state, robot_state)
          return state