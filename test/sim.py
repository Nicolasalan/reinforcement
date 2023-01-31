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

print("\033[92m\Simulation Unit Tests\033[0m")

class TestROS(unittest.TestCase):

     def setUp(self):
          current_dir = os.path.dirname(os.path.abspath(__file__))
          # navigate to the parent directory
          parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
          # navigate to the config directory
          config_dir = os.path.join(parent_dir, 'config')
          
          self.rc = Extension(config_dir)

          # Create service proxies
          self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
          self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

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


if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestROS)