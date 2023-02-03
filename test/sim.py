#! /usr/bin/env python3

from motion.utils import Extension
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import GetModelState, GetWorldProperties

import unittest
import rospy
import rostest

import os

PKG = 'motion'
NAME = 'ros'

print("\033[92mSimulation Unit Tests\033[0m")

class TestROS(unittest.TestCase):

     def setUp(self):
          rospy.init_node('test_sim_node', anonymous=True) 
          current_dir = os.path.dirname(os.path.abspath(__file__))
          # navigate to the parent directory
          parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
          # navigate to the config directory
          config_dir = os.path.join(parent_dir, 'config')
          
          self.rc = Extension(config_dir)

          # Create service proxies
          self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
          self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
          self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
          self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
          self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)

     def test_reset_simulation(self):
          # Call the reset simulation service
          success = self.reset.call()
          # Check that the service call was successful
          self.assertTrue(success, "Failed to reset simulation")
     
     def test_reset_world(self):
          # Call the reset simulation service
          success = self.reset_proxy.call()
          # Check that the service call was successful
          self.assertTrue(success, "Failed to reset world")
        
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

     def test_get_world_properties(self):
          rospy.wait_for_service("gazebo/get_world_properties")
          get_world_properties = rospy.ServiceProxy("gazebo/get_world_properties", GetWorldProperties)
          world_properties = get_world_properties()
          self.assertEqual(world_properties.sim_time, world_properties.sim_time, "Getting world properties failed")

     def test_get_model_state(self):
          rospy.wait_for_service("gazebo/get_model_state")
          get_model_state = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
          model_state = get_model_state("target", "")
          self.assertEqual(model_state.success, True, "Getting model state failed")

if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestROS)