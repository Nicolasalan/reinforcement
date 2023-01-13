#! /usr/bin/env python3

from motion.utils import Extension
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 

import unittest
import subprocess
import rospy
import rostest

import yaml
import os
import time

PKG = 'motion'
NAME = 'ros'

print("\033[92m\nROS Unit Tests\033[0m")

# folder to load config file
CONFIG_PATH = rospy.get_param('~config_path')

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        param = yaml.safe_load(file)

    return param

param = load_config("main_config.yaml")

class TestROS(unittest.TestCase):

     def setUp(self):
          self.rc = Extension()
          self.success = False
          self.rate = rospy.Rate(1)

          # Create service proxies
          self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
          self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

     def callback(self, msg):
          self.success = msg.angular.z and msg.angular.z == 1

     def test_publish_cmd_vel(self):
          # Test function for the publish_cmd_vel function.   
          test_sub = rospy.Subscriber("/cmd_vel", Twist, self.callback)
          self.rc.cmd_vel.angular.z = 1
          self.rc.publish_cmd_vel()
          timeout_t = time.time() + 1.0  # 10 seconds
          while not rospy.is_shutdown() and not self.success and time.time() < timeout_t:
               time.sleep(0.1)
          self.assert_(self.success)

     def test_subscribe_odom(self):
          # Try to receive a message from the /odom topic
          msg = rospy.wait_for_message(param["topic_odom"], Odometry, timeout=1.0)
          self.rate.sleep()
          # Verify that the message was received
          self.assertIsNotNone(msg, "Failed to receive message from /odom topic")

     def test_subscribe_scan(self):
          # Try to receive a message from the /scan topic
          msg = rospy.wait_for_message(param["topic_scan"], LaserScan, timeout=1.0)
          self.rate.sleep()
          # Verify that the message was received
          self.assertIsNotNone(msg, "Failed to receive message from /scan topic")  

     def test_reset_simulation(self):
          # Call the reset simulation service
          success = self.reset.call()
          # Check that the service call was successful
          self.assertTrue(success, "Failed to reset simulation")
        
     def test_pause_physics(self):
          # Call the pause physics service
          success = self.pause.call()
          # Check that the service call was successful
          self.assertTrue(success, "Failed to pause physics")
          
     def test_unpause_physics(self):
          # Call the unpause physics service
          success = self.unpause.call()
          # Check that the service call was successful
          self.assertTrue(success, "Failed to unpause physics")

     # TODO: model, map

if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestROS)