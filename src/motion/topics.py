
#! /usr/bin/env python3

from geometry_msgs.msg import Twist
import numpy as np
import math
import yaml
import os
import rospy

class Mensage():
     def __init__(self, CONFIG_PATH):       

          self.CONFIG_PATH = CONFIG_PATH
          param = self.load_config("main_config.yaml")

          self.vel_publisher = rospy.Publisher(param["topic_cmd"], Twist, queue_size=1)
          self.cmd_vel = Twist()
          self.ctrl_c = False
          self.rate = rospy.Rate(1)
          self.cmd = param["topic_cmd"]

     def publish_cmd_vel(self): 
          """Publishes a command velocity message to control the robot's movement."""

          while not self.ctrl_c:
               connections = self.vel_publisher.get_num_connections()
               if connections > 0:
                    self.vel_publisher.publish(self.cmd_vel)
                    break
               else:
                    self.rate.sleep()

     def load_config(self, config_name):
          with open(os.path.join(self.CONFIG_PATH, config_name)) as file:
               param = yaml.safe_load(file)

          return param

     def shutdownhook(self):
          """Shutdown hook for the node."""

          rospy.is_shutdown()