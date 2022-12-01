#!/usr/bin/env python3

import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

diagonal_dis = math.sqrt(2) * (3.6 + 3.8) # calcular a distancia diagonal do mapa
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')


class Env():
    def __init__(self, is_training):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10) # publicar a velocidade do robo
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry) # receber a odometria do robo
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty) # resetar o ambiente
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty) # despausar o ambiente
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty) # pausar o ambiente
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel) # spawnar o objetivo
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel) # deletar o objetivo
        self.past_distance = 0.
        if is_training:
            self.threshold_arrive = 0.2 # distancia minima para considerar que o robo chegou no objetivo
        else:
            self.threshold_arrive = 0.4 

    # pegar a distancia do robo para o objetivo
    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y) # distancia euclidiana
        self.past_distance = goal_distance # distancia euclidiana = distancia do robo para o objetivo

        return goal_distance # retornar a distancia do robo para o objetivo

    # pegar a odometria do robo
    def getOdometry(self, odom):
        self.position = odom.pose.pose.position # pegar a posicao do robo
        orientation = odom.pose.pose.orientation # pegar a orientacao do robo
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w # pegar a orientacao do robo em quaternions
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z)))) # pegar o yaw do robo

        if yaw >= 0: # se o yaw for maior ou igual a 0
             yaw = yaw # yaw = yaw
        else: # se o yaw for menor que 0
             yaw = yaw + 360 # yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1) # distancia relativa em x
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1) # distancia relativa em y 

        # Calcule o ângulo entre o robô e o alvo
        if rel_dis_x > 0 and rel_dis_y > 0: # se a distancia relativa em x for maior que 0 e a distancia relativa em y for maior que 0
            theta = math.atan(rel_dis_y / rel_dis_x) # calcular o theta
        elif rel_dis_x > 0 and rel_dis_y < 0: # se a distancia relativa em x for maior que 0 e a distancia relativa em y for menor que 0
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0: # se a distancia relativa em x for menor que 0 e a distancia relativa em y for menor que 0
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0: # se a distancia relativa em x for menor que 0 e a distancia relativa em y for maior que 0
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0: # se a distancia relativa em x for igual a 0 e a distancia relativa em y for maior que 0
            theta = 1 / 2 * math.pi 
        elif rel_dis_x == 0 and rel_dis_y < 0: # se a distancia relativa em x for igual a 0 e a distancia relativa em y for menor que 0
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0: # se a distancia relativa em y for igual a 0 e a distancia relativa em x for maior que 0
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2) # calcular o theta relativo

        diff_angle = abs(rel_theta - yaw) # calcular a diferenca de angulo entre o theta relativo e o yaw

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2) # arredondar a diferenca de angulo
        else:
            diff_angle = round(360 - diff_angle, 2) # arredondar a diferenca de angulo

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    # pegar o estado do robo
    def getState(self, scan):
        scan_range = [] # scan 
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2 # distancia minima
        done = False # se o robo chegou no objetivo
        arrive = False

        for i in range(len(scan.ranges)): # para cada leitura do laser
            if scan.ranges[i] == float('Inf'): # se a leitura for infinita
                scan_range.append(3.5) # adicionar 3.5 no scan 
            elif np.isnan(scan.ranges[i]): # se a leitura for nan
                scan_range.append(0) # adicionar 0
            else:
                scan_range.append(scan.ranges[i]) # adicionar a leitura

        if min_range > min(scan_range) > 0: # se a distancia minima for maior que a menor distancia do scan e a menor distancia do scan for maior que 0
            done = True # o robo chegou no objetivo

        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y) # distancia euclidiana
        if current_distance <= self.threshold_arrive: # se a distancia euclidiana for menor ou igual ao threshold de chegada
            # done = True
            arrive = True # o robo chegou no objetivo

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive # retornar o scan, a distancia euclidiana, o yaw, o theta relativo, a diferenca de angulo, se o robo chegou no objetivo e se o robo chegou no objetivo

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
                goal_urdf = open(goal_model_dir, "r").read()
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
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]

        for pa in past_action:
            state.append(pa)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive)

        return np.asarray(state), reward, done, arrive

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            rospy.loginfo("gazebo/reset_simulation service call failed")

        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            self.goal_position.position.x = random.uniform(-3.6, 3.6)
            self.goal_position.position.y = random.uniform(-3.6, 3.6)

            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            rospy.loginfo("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]

        state.append(0)
        state.append(0)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)