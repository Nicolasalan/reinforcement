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

# calcular a distância diagonal do robô
diagonal_dis = math.sqrt(2) * (3.6 + 3.8)

# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("main_config.yaml")

class Env():
     def __init__(self, num_scan_ranges=10, min_range=0.2):

          self.num_scan_ranges = num_scan_ranges
          self.n_step = 0
          self.min_range = min_range

          self.position = Pose() # posição do robô
          self.goal_position = Pose() # posição do alvo
          self.goal_position.position.x = 0.0 # posição x do alvo
          self.goal_position.position.y = 0.0 # posição y do alvo

          # definir o diretório do robô, alvo e mundo
          self.goal_model_dir = config["target"]

          ##### publicacoes e assinaturas do ROS #####
          self.pub_cmd_vel = rospy.Publisher(config["topic_cmd"], Twist, queue_size=10) # publicar a velocidade do robô
          self.sub_odom = rospy.Subscriber(config["topic_odom"], Odometry, self.getOdometry) # receber a posição do robô

          ##### servicos do ROS #####
          self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
          self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
          self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
          self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
          self.past_distance = 0.0 # distância do alvo no passado

          # definir o estado inicial
          self.threshold_arrive = config["threshold_arrive"] # distância de chegada

     ##### funcao para pegar a distancia do alvo #####
     def getGoalDistace(self):
          # calcular a distancia do alvo em relacao ao robo
          goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
          # distancia do alvo no passado se torna a distancia do alvo atual
          self.past_distance = goal_distance
          # retornar a distancia do alvo
          return goal_distance

     ##### funcao para pegar a posicao do robo por meio do topico '/odom' #####
     def getOdometry(self, odom):
          # pegar a posicao do robo
          self.position = odom.pose.pose.position
          # pegar a orientacao do robo
          orientation = odom.pose.pose.orientation
          # converter a orientacao do robo para euler
          q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
          # converter a orientacao do robo para euler
          yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

          # se o yaw for maior que 0, entao o yaw sera o yaw atual
          if yaw >= 0:
               yaw = yaw
          # se nao for maior que 0, entao o yaw sera o yaw atual mais 360
          else:
               yaw = yaw + 360
          # relacao de distancia entre o robo e o alvo em relacao ao eixo x
          rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
          # relacao de distancia entre o robo e o alvo em relacao ao eixo y
          rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

          # Calcule o ângulo entre o robô e o alvo
          if rel_dis_x > 0 and rel_dis_y > 0:
               theta = math.atan(rel_dis_y / rel_dis_x)
          elif rel_dis_x > 0 and rel_dis_y < 0:
               theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
          elif rel_dis_x < 0 and rel_dis_y < 0:
               theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
          elif rel_dis_x < 0 and rel_dis_y > 0:
               theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
          elif rel_dis_x == 0 and rel_dis_y > 0:
               theta = 1 / 2 * math.pi
          elif rel_dis_x == 0 and rel_dis_y < 0:
               theta = 3 / 2 * math.pi
          elif rel_dis_y == 0 and rel_dis_x > 0:
               theta = 0
          else:
               theta = math.pi
          rel_theta = round(math.degrees(theta), 2)

          diff_angle = abs(rel_theta - yaw)

          if diff_angle <= 180:
               diff_angle = round(diff_angle, 2)
          else:
               diff_angle = round(360 - diff_angle, 2)

          self.rel_theta = rel_theta # relacao de angulo entre o robo e o alvo
          self.yaw = yaw # orientacao do robo
          self.diff_angle = diff_angle # diferenca de angulo entre o robo e o alvo

     def getState(self, scan):
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

          if self.min_range > min(scan_range) > 0:
               done = True

          current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
          if current_distance <= self.threshold_arrive:
               arrive = True

          return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

     def setReward(self, done, arrive):
          current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
          distance_rate = (self.past_distance - current_distance)

          reward = 500.*distance_rate
          self.past_distance = current_distance

          if done:
               reward = -100.
               self.pub_cmd_vel.publish(Twist())

          if arrive:
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
               self.goal_distance = self.getGoalDistace()
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

          state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
          state = [i / 3.5 for i in state]

          for pa in past_action:
               state.append(pa)

          state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
          reward = self.setReward(done, arrive)

          return np.asarray(state), reward, done

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

          self.goal_distance = self.getGoalDistace()
          state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
          state = [i / 3.5 for i in state]

          state.append(0)
          state.append(0)

          state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

          return np.asarray(state)