#! /usr/bin/env python3

from motion.utils import Extension
import unittest
import rosunit

import os

PKG = 'motion'
NAME = 'library'

print("\033[92mLibrary Unit Tests\033[0m")

class TestLibrary(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        config_dir = os.path.join(parent_dir, 'config')

        self.rc = Extension(config_dir)
         
        # examples of data entries
        self.path_goals = [(-0.486865, 6.297588), (-1.036671, 5.001958), (-1.581001, 6.032743), (-1.972984, 5.123956), 
            (-2.759404, 6.201657), (-2.999777, 4.887653), (-3.68207, 6.087976), (-3.676394, 5.217483)]
        self.path_target = [(-0.486865, 6.297588, 6.65), (-1.036671, 5.001958, 4.232), (-1.581001, 6.032743, 3.3), (-1.972984, 5.123956, 32.5), 
            (-2.759404, 6.201657, 5), (-2.999777, 4.887653, 32), (-3.68207, 6.087976, 43), (-3.676394, 5.217483, 3)]
        self.scan_data = (6.936707019805908, 5.523979187011719, 5.137536525726318, 4.584518909454346, 4.104661464691162, 
            4.121588230133057, 4.140231132507324, 4.16062593460083)  
        self.scan = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  
            0.53439379, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.gaps = [[-1.5707963267948966, -1.4660765716752369], [-1.4660765716752369, -1.3613568165555772], 
            [-1.3613568165555772, -1.2566370614359175], [-1.2566370614359175, -1.1519173063162578]]
        self.environment = 30

    """
    Test: Angle of the robot in relation to the target
    ======
        Input (float): odom x, odom y, goal x, goal y, yaw
        Output (float): theta
    """
    def test_angles(self):

        resp = self.rc.angles(1.0, 1.0, 2.0, 2.0, 0.0)
        self.assertEquals(resp, 0.7853981633974484, "0.7853981633974484!=0.7853981633974484")
        self.rc.shutdownhook()

    """
    Test: Distance between robot and target
    ======
        Input (float): odom x, odom y, goal x, goal y
        Output (float): distance
    """
    def test_distance(self):
     
        resp = self.rc.distance_to_goal(1.0, 2.0, 9.0, 0.0)
        self.assertEquals(resp, 8.246211251235321, "8.246211251235321!=8.246211251235321")
        self.rc.shutdownhook()

    """
    Test: Agent reward
    ======
        Input (boolean, boolean, array, float): Done, collision, action (linear, angular), minimum laser distance
        Output (float): reward
    """
    def test_get_reward(self):
     
        resp = self.rc.get_reward(False, False, [0.5205170887624098, 1.0], 0.48969200253486633)
        self.assertEquals(resp, -0.49489545435136195, "-0.49489545435136195!=-0.49489545435136195")
        self.rc.shutdownhook()

    """
    Test: Collision check
    ======
        Input (Array, float): scan data and minimum distance
        Output (boolean, boolean, float): done, collision, minimum laser distance
    """ 
    def test_observe_collision(self):
     
        done, collision, min_laser = self.rc.observe_collision(self.scan, 0.1)
        self.assertEquals(min_laser, 0.53439379, "0.53439379!=0.53439379")
        self.assertFalse(collision, False)
        self.rc.shutdownhook()
    
    """
    Test: target randomization
    ======
        Input (array, float, float): map points and x, y
        Output (float): x and y
    """ 
    def test_change_goal(self):
     
        x, y = self.rc.change_goal(self.path_goals)
        self.assertTrue(-10.0 <= x <= 10.0)
        self.assertTrue(-10.0 <= y <= 10.0)
        self.rc.shutdownhook()

    """
    Test: Randomization of positions x and y
    ======
        Input (array): map points
        Output (float): x and y
    """ 
    def test_random_goal(self):
     
        x, y = self.rc.random_goal(self.path_goals)
        self.assertTrue(-10.0 <= x <= 10.0)
        self.assertTrue(-10.0 <= y <= 10.0)
        self.rc.shutdownhook()
    
    """
    Test: Check position in duplicate
    ======
        Input (float): four positions, objectives and position
        Output (boolean): verification
    """ 
    def test_check_pose(self):
     
        resp = self.rc.check_pose(1.0, 2.0, 3.0, 4.0)
        self.assertTrue(resp, True)
        self.rc.shutdownhook()

    """
    Test: Array of gaps
    ======
        Input (int): environment
        Output (array): gaps
    """ 
    def test_array_gaps(self):
     
        resp = self.rc.array_gaps(30)
        self.assertEquals(resp[0][0], -1.5707963267948966, "-1.5707963267948966!=-1.5707963267948966")
        self.rc.shutdownhook()

    """
    Test: Scan range
    ======
        Input (int, array, array): environment, gaps, scan data
        Output (array): scan range
    """ 
    def test_scan_rang(self):
     
        resp = self.rc.scan_rang(30, self.gaps, self.scan_data)
        self.assertEquals(resp[0], 10.0, "10.0!=10.0")
        self.rc.shutdownhook()

if __name__ == '__main__':
    rosunit.unitrun(PKG, NAME, TestLibrary)