#! /usr/bin/env python3

import unittest
import rosunit

PKG = 'vault'
NAME = 'package'

print("\033[92mPackage Unit Tests\033[0m")

class TestPackage(unittest.TestCase):

     def setUp(self):
          pass 

     def test_yaml_import(self):
          try:
               import yaml
          except ImportError:
               self.fail("Could not import yaml")

     def test_rospy_import(self):
          try:
               import rospy
          except ImportError:
               self.fail("Could not import rospy")

     def test_os_import(self):
          try:
               import os
          except ImportError:
               self.fail("Could not import os") 

     def test_torch_import(self):
          try:
               import torch
          except ImportError:
               self.fail("Could not import torch")  

     def test_random_import(self):
          try:
               import random
          except ImportError:
               self.fail("Could not import random") 

     def test_numpy_import(self):
          try:
               import numpy
          except ImportError:
               self.fail("Could not import numpy")  

     def test_collections_import(self):
          try:
               import collections
          except ImportError:
               self.fail("Could not import collections")

     def test_tf_import(self):
          try:
               import tf
          except ImportError:
               self.fail("Could not import tf")

     def test_math_import(self):
          try:
               import math
          except ImportError:
               self.fail("Could not import math")

     def test_squaternion_import(self):
          try:
               import squaternion
          except ImportError:
               self.fail("Could not import squaternion")

     def test_copy_import(self):
          try:
               import copy
          except ImportError:
               self.fail("Could not import copy")

     def test_geometry_msgs_import(self):
          try:
               from geometry_msgs.msg import Twist
          except ImportError:
               self.fail("Could not import geometry_msgs")

     def test_sensor_msgs_import(self):
          try:
               from sensor_msgs.msg import LaserScan
          except ImportError:
               self.fail("Could not import sensor_msgs")

     def test_nav_msgs_import(self):
          try:
               from nav_msgs.msg import Odometry
          except ImportError:
               self.fail("Could not import nav_msgs")

     def test_gazebo_msgs_import(self):
          try:
               from gazebo_msgs.msg import ModelState
          except ImportError:
               self.fail("Could not import gazebo_msgs")

     def test_std_srvs_import(self):
          try:
               from std_srvs.srv import Empty
          except ImportError:
               self.fail("Could not import std_srvs")      

     def test_psutil_import(self):
          try:
               import psutil
          except ImportError:
               self.fail("Could not import psutil")   

     def test_time_import(self):
          try:
               import time
          except ImportError:
               self.fail("Could not import time") 

     def test_opencv_import(self):
          try:
               import cv2
          except ImportError:
               self.fail("Could not import cv2") 

     def test_trimesh_import(self):
          try:
               import trimesh
          except ImportError:
               self.fail("Could not import trimesh") 

     def test_OccupancyGrid_import(self):
          try:
               from nav_msgs.msg import OccupancyGrid
          except ImportError:
               self.fail("Could not import OccupancyGrid")

if __name__ == '__main__':
    rosunit.unitrun(PKG, NAME, TestPackage)