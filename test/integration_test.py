#! /usr/bin/env python3

from motion.environment import Env
import unittest

class CaseA(unittest.TestCase):

    def setUp(self):
        self.rc = Env()

    def test_state(self):

        resp = self.state()
        self.assertEquals()
        self.rc.shutdownhook()

class CaseB(unittest.TestCase):
     
    def setUp(self):
        self.rc = Env()

    def test_reset(self):

        resp = self.rc.reset()
        self.assertTrue()
        self.rc.shutdownhook()

class CaseA(unittest.TestCase):
     
    def setUp(self):
        self.rc = Env()

    def test_state(self):

        resp = self.state()
        self.assertEquals()
        self.rc.shutdownhook()

class CaseC(unittest.TestCase):
     
    def setUp(self):
        self.rc = Env()

    def test_reward(self):

        resp = self.rc.reward()
        self.assertTrue()
        self.rc.shutdownhook()

class TestSuite(unittest.TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(CaseA())
        self.addTest(CaseB())
        self.addTest(CaseC())