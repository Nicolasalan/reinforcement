
#! /usr/bin/env python3

from gazebo_msgs.srv import GetWorldProperties, GetModelState
from gazebo_msgs.msg import ModelState
import numpy as np
import math
import yaml
import os
import rospy
import time
import random

class Extension():
     def __init__(self, CONFIG_PATH):       

          self.CONFIG_PATH = CONFIG_PATH
          param = self.load_config("config.yaml")

          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
          self.get_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
          self.robot = param["robot"]

          self.state_dim = param["environment_dim"] + param["robot_dim"]
          self.action_dim = param["action_dim"]
          self.cmd = param["topic_cmd"]
          self.odom = param["topic_odom"]
          self.scan = param["topic_scan"]
          self.goal_reached_dist = param["goal_reached_dist"]
          self.collision_dist = param["collision_dist"] 
          self.time_delta = param["time_delta"]  

     def angles(self, odom_x, odom_y, goal_x, goal_y, angle):
          """Calculate the relative angle between the robots heading and heading toward the goal."""

          skew_x = goal_x - odom_x
          skew_y = goal_y - odom_y
          dot = skew_x * 1 + skew_y * 0
          mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
          mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
          if mag1 * mag2 == 0:
               beta = 0
          else:
               beta = math.acos(dot / (mag1 * mag2))
          if skew_y < 0:
               if skew_x < 0:
                    beta = -beta
               else:
                    beta = 0 - beta
          theta = beta - angle
          if theta > np.pi:
               theta = np.pi - theta
               theta = -np.pi - theta
          if theta < -np.pi:
               theta = -np.pi - theta
               theta = np.pi - theta

          return theta

     def distance_to_goal(self, odom_x, odom_y, goal_x, goal_y):
          """Calculate the distance between the robot and the goal."""

          distance = np.linalg.norm([odom_x - goal_x, odom_y - goal_y]) # vector length

          return distance

     def path_goal(self, path_waypoints):
          """Load the waypoints from the yaml file."""
          
          goals = []
          list = []

          with open(path_waypoints) as f:
               data = yaml.safe_load(f)
               for i in data:
                    list.append(i['position'])

               for i in list:
                    str_x = str(i[0]).strip('[]')
                    str_y = str(i[1]).strip('[]')
                    x = float(str_x)
                    y = float(str_y)
                    goals.append((x, y))

          return goals

     def path_target(self, path_waypoints):
          """Load the waypoints from the yaml file."""
          
          goals = []
          list = []

          with open(path_waypoints) as f:
               data = yaml.safe_load(f)
               for i in data:
                    list.append(i['position'])

               for i in list:
                    str_x = str(i[0]).strip('[]')
                    str_y = str(i[1]).strip('[]')
                    str_yaw = str(i[2]).strip('[]')
                    
                    x = float(str_x)
                    y = float(str_y)
                    yaw = float(str_yaw)
                    goals.append((x, y, yaw))

          return goals

     def path_objects(self, path_waypoints):
          """Load the waypoints from the yaml file."""
          
          goals = []
          list = []

          with open(path_waypoints) as f:
               data = yaml.safe_load(f)
               for i in data:
                    list.append(i['position'])

               for i in list:
                    str_x = str(i[0]).strip('[]')
                    str_y = str(i[1]).strip('[]')
                    str_yaw = str(i[2]).strip('[]')
                    
                    x = float(str_x)
                    y = float(str_y)
                    yaw = float(str_yaw)
                    goals.append((x, y, yaw))

          return goals
     
     def get_reward(self, target, collision, action, min_laser):
          """Agent reward function."""

          if target:
               return 100.0
          elif collision:
               return -100.0
          else:
               # This gives an additional negative reward if the robot is closer to any obstacle than 1 meter. Using this 'repulsion' causes the robot to get more tired of obstacles in general and go around them with a greater go.
               r3 = lambda x: 1 - x if x < 1 else 0.0
               return (action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2)

     def observe_collision(self, laser_data, collision_dist):
          """Detect a collision from laser data."""

          min_laser = min(laser_data)
          if min_laser < collision_dist:
               return True, True, min_laser
          return False, False, min_laser

     def array_gaps(self, environment_dim):
          """Retorna uma matriz de intervalos representando lacunas em um determinado ambiente.."""

          gaps = [[-np.pi, -np.pi + 2 * np.pi / environment_dim]]
          for m in range(environment_dim - 1):
               gaps.append([gaps[m][1], gaps[m][1] + 2 * np.pi / environment_dim])
               gaps[-1][-1] += 0.01  # add a small offset to the last gap to avoid overlap

          return gaps

     def scan_rang(self, environment_dim, scan_data):
          """Returns an array of the minimum distances from the laser scan data to the gaps in a given environment."""

          laser_data = np.ones(environment_dim) * 10
          for i in range(len(scan_data.ranges)):
               dist = scan_data.ranges[i]
               if not np.isnan(dist):
                    # calculate the index of the laser data array that corresponds to the current angle
                    angle = scan_data.angle_min + i * scan_data.angle_increment
                    index = int((angle + np.pi/2) / (np.pi / environment_dim))
                    # make sure that the index is within the bounds of the laser data array
                    if index >= 0 and index < environment_dim:
                         if dist < laser_data[index]:
                              laser_data[index] = dist
          return laser_data
          
     def range(self, scan):
          """Returns an array of the minimum distances from the laser scan data"""

          scan_range = []
          for i in range(len(scan.ranges)):
               if scan.ranges[i] == float('Inf'):
                    scan_range.append(30.0)
               elif np.isnan(scan.ranges[i]):
                    scan_range.append(0)
               else:
                    scan_range.append(scan.ranges[i])

          return np.array(scan_range)

     def shutdownhook(self):
          """Shutdown hook for the node."""
          rospy.is_shutdown()

     def load_config(self, config_name):
          with open(os.path.join(self.CONFIG_PATH, config_name)) as file:
               param = yaml.safe_load(file)

          return param

     def select_poses(self, poses):
          """Select two random poses from the list of poses."""

          if len(poses) < 2:
               raise ValueError("The 'poses' list must have at least two elements")

          index_robot = int(round(np.random.uniform(0, len(poses))))
          index_target = int(round(np.random.uniform(0, len(poses))))
          while index_robot == index_target:
               index_target = int(round(np.random.uniform(0, len(poses))))

          return poses[index_robot], poses[index_target]

     def select_random_poses(self, poses, percentage):
          """Select a random percentage of poses from the list of poses."""

          n = int(len(poses) * (percentage))

          return random.sample(poses, n)

     def random_near_obstacle(self, state, count_rand_actions, random_action, add_noise=True):
          """Select a random action near an obstacle."""

          if add_noise: 

               if (np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

               if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
               
               return action, count_rand_actions, random_action

