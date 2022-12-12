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

          self.num_scan_ranges = param["num_scan_ranges"]
          self.min_range = param["min_range"]

          self.last_odom = None
          self.odom_x = 0
          self.odom_y = 0

          self.goal_x = 1
          self.goal_y = 0.0

          self.diagonal = math.sqrt(2) * (3.6 + 3.8)
          self.goal_model = '/ws/src/motion/models/target.sdf'

          ##### publicacoes e assinaturas do ROS #####
          self.pub_cmd_vel = rospy.Publisher(param["topic_cmd"], Twist, queue_size=10) # publicar a velocidade do robô
          self.odom = rospy.Subscriber(param["topic_odom"], Odometry, self.odom_callback, queue_size=1) # receber a posição do robô

          ##### servicos do ROS #####
          self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          #self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
          self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
          self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
          self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
          self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
          self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
          self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
          self.past_distance = 0.0

          # definir o estado inicial
          self.threshold_target = param["threshold_target"] # distância de chegada

          self.goals = []
          self.goals_id = 0
          list = []

          with open(param["waypoints"]) as f:
               data = yaml.safe_load(f)
               for i in data:
                    list.append(i['position'])

               for i in list:
                    str_x = str(i[0]).strip('[]')
                    str_y = str(i[1]).strip('[]')
                    x = float(str_x)
                    y = float(str_y)
                    # add x and y to goals
                    self.goals.append((x, y))

     # funcao para pegar a posicao do robo por meio do topico '/odom' 
     def odom_callback(self, od_data):
          self.last_odom = od_data
     
     def check_scan_range(self, scan, num_scan_ranges):
          scan_range = []
          cof = (len(scan.ranges) / (num_scan_ranges - 1))  # calcula o coeficiente de escala
          for i in range(0, num_scan_ranges): 
               n_i = math.ceil(i*cof - 1) # calcula o indice do scan
               if n_i < 0: 
                    n_i = 0 
               if cof == 1:
                    n_i = i 
               if scan.ranges[n_i] == float('Inf'): 
                    scan_range.append(3.5) 
               elif np.isnan(scan.ranges[n_i]): 
                    scan_range.append(0) 
               else:
                    scan_range.append(scan.ranges[n_i]) 
          
          return scan_range

     def setReward(self, done, target):
          # Calculate robot heading from odometry data
          self.odom_x = self.last_odom.pose.pose.position.x
          self.odom_y = self.last_odom.pose.pose.position.y

          distance = math.hypot(self.odom_x - self.goal_x, self.odom_y - self.goal_y)

          distance_rate = (self.past_distance - distance)

          reward = 500.*distance_rate
          self.past_distance = distance

          if done: # se o robô colidir com algum obstáculo
               reward = -100.
               self.pub_cmd_vel.publish(Twist())

          if target: # se o robô chegar ao alvo
               reward = 120.
               self.pub_cmd_vel.publish(Twist())

               try:
                    # randomiza o target pelo mundo
                    print("Randomizing target position")

                    points = self.goals
                    x, y = 0, 0
                    rand = int(round(random.uniform(0, len(points))))
                    for i in range(len(points)):
                         if i == rand:
                              x, y = points[i][0], points[i][1] 

                    print("Goal x: ", x, "Goal y: ", y)

                    box_state = ModelState()
                    box_state.model_name = "target"
                    box_state.pose.position.x = x
                    box_state.pose.position.y = y
                    box_state.pose.orientation.x = 0.0
                    box_state.pose.orientation.y = 0.0
                    box_state.pose.orientation.z = 0.0
                    box_state.pose.orientation.w = 1.0
                    self.set_state.publish(box_state)
                    print("Target randomized")
                    
                    self.goal_x = x
                    self.goal_y = y
               
               except (rospy.ServiceException) as e:
                    print("/gazebo/failed to build the target")
               rospy.wait_for_service('/gazebo/unpause_physics')

               self.odom_x = self.last_odom.pose.pose.position.x
               self.odom_y = self.last_odom.pose.pose.position.y

               # Calculate distance to the goal from the robot
               distance = np.linalg.norm(
                    [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
               )
               self.past_distance = distance
               target = False

          return reward

     def state(self, scan):
          done = False
          target = False

          # Calculate robot heading from odometry data
          self.odom_x = self.last_odom.pose.pose.position.x
          self.odom_y = self.last_odom.pose.pose.position.y
          quaternion = Quaternion(
               self.last_odom.pose.pose.orientation.w,
               self.last_odom.pose.pose.orientation.x,
               self.last_odom.pose.pose.orientation.y,
               self.last_odom.pose.pose.orientation.z,
          )

          yaw = round(math.degrees(math.atan2(2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y), 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))), 2)

          if yaw >= 0:
               yaw = yaw
          else:
               yaw = yaw + 360

          distance = math.hypot(self.odom_x - self.goal_x, self.odom_y - self.goal_y)

          skew_x = self.goal_x - self.odom_x
          skew_y = self.goal_y - self.odom_y

          # Calculate the angle between robot and target
          if skew_x > 0 and skew_y > 0:
               theta = math.atan(skew_y / skew_x)
          elif skew_x > 0 and skew_y < 0:
               theta = 2 * math.pi + math.atan(skew_y / skew_x)
          elif skew_x < 0 and skew_y < 0:
               theta = math.pi + math.atan(skew_y / skew_x)
          elif skew_x < 0 and skew_y > 0:
               theta = math.pi + math.atan(skew_y / skew_x)
          elif skew_x == 0 and skew_y > 0:
               theta = 1 / 2 * math.pi
          elif skew_x == 0 and skew_y < 0:
               theta = 3 / 2 * math.pi
          elif skew_y == 0 and skew_x > 0:
               theta = 0
          else:
               theta = math.pi
          thetas = round(math.degrees(theta), 2)

          diff = abs(thetas - yaw)

          if diff <= 180:
               diff = round(diff, 2)
          else:
               diff = round(360 - diff, 2)

          scan_range = self.check_scan_range(scan, self.num_scan_ranges)

          if self.min_range > min(scan_range) > 0: # se o robô colidir com algum obstáculo
               done = True
          
          # Detect if the goal has been reached and give a large positive reward
          if distance <= self.threshold_target:
               target = True

          return scan_range, distance, yaw, thetas, diff, done, target

     def step(self, action):
          # Publish the robot action
          vel_cmd = Twist()
          vel_cmd.linear.x = action[0]
          vel_cmd.angular.z = action[1]
          self.pub_cmd_vel.publish(vel_cmd)

          past = np.array([0., 0.])
          data = None
          while data is None:
               try:
                    data = rospy.wait_for_message(param["topic_scan"], LaserScan, timeout=5)
               except:
                    print("Waiting for data")
                    pass
          
          min_laser, distance, yaw, thetas, diff, done, target = self.state(data)
          states = [i / 3.5 for i in min_laser] # normalizar os dados de entrada

          for action in past: # adicionar a ação anterior ao estado
               states.append(action)

          states = states + [distance / self.diagonal, yaw / 360, thetas / 360, diff / 180]
          reward = self.setReward(done, target)

          return np.asarray(states), reward, done, target

     def reset(self):
          print("Resetting environment")

          rospy.wait_for_service('gazebo/reset_simulation')
          try:
               self.reset_proxy()
          except (rospy.ServiceException) as e:
               print("gazebo/reset_simulation service call failed")

          print("Resetting robot")
          # Build the target
          try:
               # randomiza o target pelo mundo
               print("Randomizing target position")

               points = self.goals
               x, y = 0, 0
               rand = int(round(random.uniform(0, len(points))))
               for i in range(len(points)):
                    if i == rand:
                         x, y = points[i][0], points[i][1] 

               print("Goal x: ", x, "Goal y: ", y)

               box_state = ModelState()
               box_state.model_name = "target"
               box_state.pose.position.x = x
               box_state.pose.position.y = y
               box_state.pose.orientation.x = 0.0
               box_state.pose.orientation.y = 0.0
               box_state.pose.orientation.z = 0.0
               box_state.pose.orientation.w = 1.0
               self.set_state.publish(box_state)
               print("Target randomized")
               
               self.goal_x = x
               self.goal_y = y

          except (rospy.ServiceException) as e:
               print("/gazebo/failed to build the target")
          rospy.wait_for_service('/gazebo/unpause_physics')
          
          print("Unpausing physics")
          data = None
          while data is None:
               try:
                    data = rospy.wait_for_message(param["topic_scan"], LaserScan, timeout=5)
               except:
                    print("Waiting for data")
                    pass
          
          print("Odom received")
          self.odom_x = self.last_odom.pose.pose.position.x
          self.odom_y = self.last_odom.pose.pose.position.y

          # Calculate distance to the goal from the robot
          distance = np.linalg.norm(
               [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
          )
          self.past_distance = distance

          states, distance, yaw, thetas, diff, done, target = self.state(data)
          states = [i / 3.5 for i in states]

          states.append(0)
          states.append(0)

          print("Passed states")

          states = states + [distance / self.diagonal, yaw / 360, thetas / 360, diff / 180]

          return np.asarray(states)

     