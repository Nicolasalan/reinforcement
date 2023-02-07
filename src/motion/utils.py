
#! /usr/bin/env python3

from gazebo_msgs.srv import GetWorldProperties, GetModelState
from gazebo_msgs.msg import ModelState
import numpy as np
import math
import yaml
import os
import rospy
import time

class Extension():
     def __init__(self, CONFIG_PATH):       

          self.CONFIG_PATH = CONFIG_PATH
          param = self.load_config("config.yaml")

          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
          self.get_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
          self.robot = param["robot"]

          self.state_dim = param["environment_dim"] + param["robot_dim"]
          self.action_dim = param["action_dim"]
          self.action_linear_max = param["action_linear_max"]
          self.action_angular_max = param["action_angular_max"]
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

          distance = np.linalg.norm([odom_x - goal_x, odom_y - goal_y])

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

     def random_goal(self, goals):
          """Select a random goal from the list of waypoints."""

          points = goals
          x, y = 0, 0
          rand = int(round(np.random.uniform(0, len(points))))
          for i in range(len(points)):
               if i == rand:
                    x, y = points[i][0], points[i][1] 
          
          return x, y

     def get_reward(self, target, collision, action, min_laser):
          """Agent reward function."""

          if target:
               return 100.0
          elif collision:
               return -100.0
          else:
               r3 = lambda x: 1 - x if x < 1 else 0.0
               return (action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2)

     def observe_collision(self, laser_data, collision_dist):
          """Detect a collision from laser data."""

          min_laser = min(laser_data)
          if min_laser < collision_dist:
               return True, True, min_laser
          return False, False, min_laser

     def change_goal(self, goals):
          """Select a random goal from the list of waypoints."""

          points = goals
          _x, _y = 0.0, 0.0
          rand = int(round(np.random.uniform(0, len(points))))
          for i in range(len(points)):
               if i == rand:
                    _x, _y = points[i][0], points[i][1] 
          
          x = _x
          y = _y

          return x, y

     def check_pose(self, x1, y1, x2, y2):
          """checks that the position is not in conflict with another position."""
          
          if x1 == x2 and y1 == y2:
               return False
          else:
               return True

     def array_gaps(self, environment_dim):
          """Retorna uma matriz de intervalos representando lacunas em um determinado ambiente.."""

          gaps = [[-np.pi / 2, -np.pi / 2 + np.pi / environment_dim]]
          for m in range(environment_dim - 1):
               gaps.append(
                    [gaps[m][1], gaps[m][1] + np.pi / environment_dim]
               )

          return gaps

     def scan_rang(self, environment_dim, gaps, data):
          """Returns an array of the minimum distances from the laser scan data to the gaps in a given environment."""

          scan_data = np.ones(environment_dim) * 10
          for point in data:
               dot = point * 1
               mag1 = math.sqrt(point ** 2)
               mag2 = 1
               beta = math.acos(dot / (mag1 * mag2)) * np.sign(0)
               dist = math.sqrt(point ** 2)

               for j in range(len(gaps)):
                    if gaps[j][0] <= beta < gaps[j][1]:
                         scan_data[j] = min(scan_data[j], dist)
                         break

          return scan_data

     def range(self,  scan):
          """Returns an array of the minimum distances from the laser scan data"""

          scan_range = []
          for i in range(len(scan.ranges)):
               if scan.ranges[i] == float('Inf'):
                    scan_range.append(3.5)
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

     def randomize_objects(self):
          rospy.wait_for_service("gazebo/get_world_properties")
          get_world_properties = rospy.ServiceProxy("gazebo/get_world_properties", GetWorldProperties)

          model_poses = []

          response = get_world_properties()
          model_names = response.model_names
          
          model_names.remove(self.robot)
          model_names.remove('target')

          for model_name in model_names:
               rospy.wait_for_service("gazebo/get_model_state")
               try:
                    response = self.get_pose(model_name, "")
                    x = response.pose.position.x
                    y = response.pose.position.y
                    z = response.pose.position.z
                    model_poses.append((x, y, z))
               except rospy.ServiceException as e:
                    pass

          poses_copy = model_poses.copy()

          # Set each model to its new randomized position
          for name in model_names:
               state = ModelState()
               state.model_name = name
               random_pose = np.random.choice(poses_copy)
               state.pose.position.x, state.pose.position.y, state.pose.position.z = random_pose
               poses_copy.remove(random_pose)
               self.set_state.publish(state)
          
          time.sleep(self.time_delta)

          print(poses_copy)

          time.sleep(self.time_delta)


