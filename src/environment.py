#!/usr/bin/env python3

# importar bibliotecas comuns
import os
import rospy
import numpy as np
import math
from math import pi
import random
import yaml

# importar mensagens do ROS
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

from utils import getGoalDistace, getOdometry

# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("main_config.yaml")

class Env():
     def __init__(self):

          self.num_scan_ranges = config["num_scan_ranges"]
          self.n_step = 0
          self.min_range = config["min_range"]

          self.position = Pose() # posição do robô
          self.goal_position = Pose() # posição do alvo
          self.goal_position.position.x = 0.0 # posição x do alvo
          self.goal_position.position.y = 0.0 # posição y do alvo

          # definir o diretório do robô, alvo e mundo
          self.goal_model_dir = config["target"]
          # calcular a distância diagonal do robô
          self.diagonal_dis = math.sqrt(2) * (3.6 + 3.8)

          ##### publicacoes e assinaturas do ROS #####
          self.pub_cmd_vel = rospy.Publisher(config["topic_cmd"], Twist, queue_size=10) # publicar a velocidade do robô
          self.sub_odom = rospy.Subscriber(config["topic_odom"], Odometry, getOdometry) # receber a posição do robô

          self.diff_angle, self.rel_theta, self.yaw = self.sub_odom

          ##### servicos do ROS #####
          self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
          self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
          self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
          self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
          self.past_distance = 0.0 # distância do alvo no passado

          # definir o estado inicial
          self.threshold_arrive = config["threshold_arrive"] # distância de chegada

     def state(self, scan):
          scan_range = []
          yaw = self.yaw
          rel_theta = self.rel_theta
          diff_angle = self.diff_angle
          done = False
          arrive = False

          cof = (len(scan.ranges) / (self.num_scan_ranges - 1))
          for i in range(0, self.num_scan_ranges):
               n_i = math.ceil(i*cof - 1)
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

          if self.min_range > min(scan_range) > 0: # se o robô colidir com algum obstáculo
               done = True

          current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
          if current_distance <= self.threshold_arrive:
               arrive = True

          return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

     def reward(self, done, arrive):
          current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
          distance_rate = (self.past_distance - current_distance)

          reward = 500.*distance_rate
          self.past_distance = current_distance

          if done: # se o robô colidir com algum obstáculo
               reward = -100.
               self.pub_cmd_vel.publish(Twist())

          if arrive: # se o robô chegar ao alvo
               reward = 120.
               self.pub_cmd_vel.publish(Twist())
               rospy.wait_for_service('/gazebo/delete_model')
               self.del_model('target')

               # Build the target
               rospy.wait_for_service('/gazebo/spawn_sdf_model')
               try:
                    goal_urdf = open(self.goal_model_dir, "r").read()
                    target = SpawnModel
                    target.model_name = 'target'  # the same with sdf name
                    target.model_xml = goal_urdf
                    self.goal_position.position.x = random.uniform(-3.6, 3.6)
                    self.goal_position.position.y = random.uniform(-3.6, 3.6)

                    self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')

               except (rospy.ServiceException) as e:
                    print("/gazebo/failed to build the target")
               rospy.wait_for_service('/gazebo/unpause_physics')
               self.goal_distance = getGoalDistace()
               arrive = False

          return reward

     def step(self, action, past_action):
          linear_vel = action[0]
          ang_vel = action[1]

          vel_cmd = Twist()
          vel_cmd.linear.x = linear_vel 
          vel_cmd.angular.z = ang_vel
          self.pub_cmd_vel.publish(vel_cmd)

          data = None
          while data is None:
               try:
                    data = rospy.wait_for_message(config["topic_scan"], LaserScan, timeout=5)
               except:
                    pass

          states, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.state(data)
          states = [i / 3.5 for i in states] # normalizar os dados de entrada

          for pa in past_action: # adicionar a ação anterior ao estado
               states.append(pa)

          states = states + [rel_dis / self.diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
          reward = self.reward(done, arrive)

          return np.asarray(states), reward, done

     def reset(self):
          # Reset the env #
          rospy.wait_for_service('/gazebo/delete_model')
          self.del_model('target')

          rospy.wait_for_service('gazebo/reset_simulation')
          try:
               self.reset_proxy()
          except (rospy.ServiceException) as e:
               print("gazebo/reset_simulation service call failed")

          # Build the targetz
          rospy.wait_for_service('/gazebo/spawn_sdf_model')
          try:
               goal_urdf = open(self.goal_model_dir, "r").read()
               target = SpawnModel
               target.model_name = 'target'  # the same with sdf name
               target.model_xml = goal_urdf

               # randomiza o target pelo mundo
               self.goal_position.position.x = random.uniform(-3.6, 3.6)
               self.goal_position.position.y = random.uniform(-3.6, 3.6)
               self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
          except (rospy.ServiceException) as e:
               print("/gazebo/failed to build the target")
          rospy.wait_for_service('/gazebo/unpause_physics')
          data = None
          while data is None:
               try:
                    data = rospy.wait_for_message(config["topic_scan"], LaserScan, timeout=5)
               except:
                    pass

          self.goal_distance = getGoalDistace()
          states, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.state(data)
          states = [i / 3.5 for i in states]

          states.append(0)
          states.append(0)

          states = states + [rel_dis / self.diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

          return np.asarray(states)