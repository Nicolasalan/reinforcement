#! /usr/bin/env python3

from motion.agent import Agent
import unittest
import rosunit
import rospy

import yaml
import os

PKG = 'motion'
NAME = 'integration'

print("\033[92mLearning Integration Tests\033[0m")

class TestLearning(unittest.TestCase):

     def setUp(self):
          self.rc = Agent()
          self.states = [10.         10.         10.         10.         10.         10.
               10.         10.         10.         10.         10.         10.
               10.         10.         10.          0.45597157 10.         10.
               10.         10.         10.         10.         10.         10.
               10.         10.         10.         10.         10.         10.
               0.179348    1.57079633  0.3249251  -0.82032906]
          self.next_state = [10.         10.         10.         10.         10.         10.
               10.         10.         10.         10.         10.         10.
               10.         10.         10.          0.4754726  10.         10.
               10.         10.         10.         10.         10.         10.
               10.         10.         10.         10.         10.         10.
               0.179348    1.57079633  0.79990483 -0.99923811]
          self.actions = [0.43717188832626686, 0.6934132281667125]
          
                         

     def test_action(self):

          resp = self.rc.action(self.states)
          self.assertEquals(resp[0], 0.683783, "0.683783!=0.683783")
          self.rc.shutdownhook()

if __name__ == '__main__':
    rosunit.unitrun(PKG, NAME, TestLearning)