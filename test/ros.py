#! /usr/bin/env python3

from reinforcement.topics import Mensage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import unittest
import rospy
import rostest
import time

import os

PKG = 'reinforcement'
NAME = 'ros'

print("\033[92mROS Unit Tests\033[0m")

class TestROS(unittest.TestCase):

     def setUp(self):
          rospy.init_node('test_ros_node', log_level=rospy.DEBUG)
          current_dir = os.path.dirname(os.path.abspath(__file__))
          # navigate to the parent directory
          parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
          # navigate to the config directory
          config_dir = os.path.join(parent_dir, 'config')
          
          self.rc = Mensage(config_dir)
          self.param = self.rc.load_config("config.yaml")

          self.cmd = self.param["TOPIC_CMD"]
          self.odom = self.param["TOPIC_ODOM"]
          self.scan = self.param["TOPIC_SCAN"]

          self.success = False
          self.rate = rospy.Rate(1)

     def callback(self, msg):
          self.success = msg.angular.z and msg.angular.z == 1

     def map_callback(self, msg):
          self.received_map = msg

     def test_publish_cmd_vel(self):
          # Test function for the publish_cmd_vel function.   
          test_sub = rospy.Subscriber(self.cmd, Twist, self.callback)
          self.rc.cmd_vel.angular.z = 1
          self.rc.publish_cmd_vel()
          timeout_t = time.time() + 1.0  # 10 seconds
          while not rospy.is_shutdown() and not self.success and time.time() < timeout_t:
               time.sleep(0.1)
          self.assert_(self.success)

     def test_subscribe_odom(self):
          # Try to receive a message from the /odom topic
          msg = rospy.wait_for_message(self.odom, Odometry, timeout=1.0)
          self.rate.sleep()
          # Verify that the message was received
          self.assertIsNotNone(msg, "Failed to receive message from /odom topic")

     def test_subscribe_scan(self):
          # Try to receive a message from the /scan topic
          msg = rospy.wait_for_message(self.scan, LaserScan, timeout=1.0)
          self.rate.sleep()
          # Verify that the message was received
          self.assertIsNotNone(msg, "Failed to receive message from /scan topic") 
          
if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestROS)