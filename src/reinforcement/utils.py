#! /usr/bin/env python3

import numpy as np
import math
import yaml
import os
import rospy

class Extension():
     def __init__(self, CONFIG_PATH):       

          self.CONFIG_PATH = CONFIG_PATH
          param = self.load_config("config.yaml")
          self.results = param["RESULTS"]

          self.max_range = param["MAX_RANGE"]
          self.max_t = param["MAX_TIMESTEP"]

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

     def distance_to_goal(self, odom_x: float, odom_y: float, goal_x: float, goal_y: float):
          """Calculate the distance between the robot and the goal."""

          distance = math.hypot(goal_x - odom_x, goal_y - odom_y)

          return distance

     def poses(self, path_waypoints):
          """Load the waypoints from the yaml file."""
          
          goals = []
          list = []

          with open('/home/user/ws/src/reinforcement/config/pose/' + path_waypoints) as f:
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
     
     # ==== Reward Functions ==== #
     
     def get_reward(self, target, collision, action, min_laser):
          """Agent reward function."""

          if target:
               return 100.0
          elif collision:
               return -100.0
          else:
               # This gives an additional negative reward if the robot is closer to any obstacle than 1 meter. Using this 'repulsion' causes the 
               # robot to get more tired of obstacles in general and go around them with a greater go.
               r3 = lambda x: 1 - x if x < 1 else 0.0
               return (action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2)
     
     def observe_collision(self, laser_data, collision_dist):
          """Detect a collision from laser data."""
          # escrever teste aqui
          try:
               min_laser = min(laser_data)
               if min_laser < collision_dist:
                    return True, True, min_laser
               return False, False, min_laser
          except:
               return False, False, 0
          
     def range(self, scan):
          """Returns an array of the minimum distances from the laser scan data"""

          scan_range = []
          for i in range(len(scan.ranges)):
               if scan.ranges[i] == float('Inf'):
                    scan_range.append(self.max_range)
               elif np.isnan(scan.ranges[i]):
                    scan_range.append(0)
               else:
                    scan_range.append(scan.ranges[i])
     
          return np.array(scan_range)
     
     # ==== Helper Functions === #
     def shutdownhook(self):
          """Shutdown hook for the node."""
          rospy.is_shutdown()

     def load_config(self, config_name):
          with open(os.path.join('/home/user/ws/src/reinforcement/config/config.yaml')) as file:
               param = yaml.safe_load(file)

          return param

     # ==== Random Functions ==== #
     def select_poses(self, poses):
          """Select two random poses from the list of poses."""

          if len(poses) < 2:
               raise ValueError("The 'poses' list must have at least two elements")

          index_robot = int(round(np.random.uniform(0, len(poses))))
          index_target = int(round(np.random.uniform(0, len(poses))))

          # Make sure that the robot and target are not the same
          while index_robot == index_target:
               index_target = int(round(np.random.uniform(0, len(poses))))

          # Make sure that the robot and target are not too close to each other
          while index_robot >= len(poses) or index_target >= len(poses):
               index_robot = int(round(np.random.uniform(0, len(poses))))
               index_target = int(round(np.random.uniform(0, len(poses))))

          return poses[index_robot], poses[index_target]

     def random_near_obstacle(self, state, count_rand_actions, random_action, add_noise=True):
          """Select a random action near an obstacle."""
          action = np.zeros(2)

          if add_noise: 

               if (np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

               if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
                    action[0] = -1
               
               return action, count_rand_actions, random_action

          else:
               return action, count_rand_actions, random_action

     def evaluate(self, agent, env, epoch, eval_episodes=10):
          avg_reward = 0.0
          for _ in range(eval_episodes):
               state = env.reset_env()
               done = False
               for _ in range(self.max_t):
                    action = agent.action(state)
                    actions = [(action[0] + 1) / 2, action[1]]
                    state, reward, done, _ = env.step_env(actions)
                    avg_reward += reward
                    if np.any(done):                                       
                         break  

          avg_reward /= eval_episodes
          
          rospy.loginfo('# ====== Episode: ' + str(epoch) + ' Average Score: ' + str(avg_reward) + ' ====== #')
