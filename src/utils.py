
#! /usr/bin/env python3

import numpy as np
import math
import random
import yaml
import os

# folder to load config file
CONFIG_PATH = "/ws/src/motion/config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        param = yaml.safe_load(file)

    return param

param = load_config("main_config.yaml")

class Extension():
     def __init__(self):

          self.path_waypoints = param["waypoints"]
          self.collision_dist = param["collision_dist"]

     def angles(self, odom_x, odom_y, goal_x, goal_y, angle):
          # Calculate the relative angle between the robots heading and heading toward the goal
          skew_x = goal_x - odom_x
          skew_y = goal_y - odom_y
          dot = skew_x * 1 + skew_y * 0
          mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
          mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
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
          # Calculate the distance between the robot and the goal
          distance = np.linalg.norm([odom_x - goal_x, odom_y - goal_y])

          return distance

     def path_goal(self):
          # Load the waypoints from the yaml file
          goals = []
          list = []

          with open(self.path_waypoints) as f:
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

     def random_goal(self, goals):
          # Select a random goal from the list of waypoints
          points = goals
          x, y = 0, 0
          rand = int(round(random.uniform(0, len(points))))
          for i in range(len(points)):
               if i == rand:
                    x, y = points[i][0], points[i][1] 
          
          return x, y

     def get_reward(self, target, collision, action, min_laser):
          if target:
               return 100.0
          elif collision:
               return -100.0
          else:
               r3 = lambda x: 1 - x if x < 1 else 0.0
               return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

     def observe_collision(self, laser_data):
          # Detect a collision from laser data
          min_laser = min(laser_data)
          if min_laser < self.collision_dist:
               return True, True, min_laser
          return False, False, min_laser

     def change_goal(self, goals, odom_x, odom_y):
          # Select a random goal from the list of waypoints
          points = goals
          x, y = 0, 0
          rand = int(round(random.uniform(0, len(points))))
          for i in range(len(points)):
               if i == rand:
                    _x, _y = points[i][0], points[i][1] 
          
          x = odom_x + _x
          y = odom_y + _y

          return x, y

     def check_pose(self, x1, y1, x2, y2):
          if x1 == x2 and y1 == y2:
               return False
          else:
               return True
          