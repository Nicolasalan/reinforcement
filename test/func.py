#! /usr/bin/env python3

from reinforcement.utils import Extension
import unittest
import rosunit

import os

PKG = 'reinforcement'
NAME = 'library'

print("\033[92mFunction Unit Tests\033[0m")

class TestLibrary(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        config_dir = os.path.join(parent_dir, 'config')
        print(config_dir)

        self.rc = Extension(config_dir)
         
        # examples of data entries
        
        self.scan = [1.40199554, 1.60912216, 1.61471009, 1.6209532, 1.627864, 1.63545573,
            1.64374459, 1.7048229, 1.71486449, 1.72568238, 1.73730004, 1.74974227,
            1.76303685, 1.77721334, 1.79230428, 1.80834436, 1.82537198, 1.84342813,
            1.86255765, 1.8828088, 1.90423501, 1.92689204, 1.95084333, 1.97615516,
            2.00290084, 2.03116107, 2.06102157, 2.0925777, 2.12593293, 2.161201,
            2.1985054, 2.23798323, 2.23679996, 2.19318676, 2.15204215, 2.11319757,
            2.07650304, 2.04182243, 2.00902891, 1.9780091, 1.97362149, 1.9208858,
            1.83375609, 1.86972356, 1.84618092, 1.82390666, 1.80283761, 1.78291857,
            1.76409471, 1.74631965, 1.72954786, 1.71373725, 1.69885135, 1.68485391,
            1.67171431, 1.65940094, 1.64788854, 1.63714981, 1.62716186, 1.61790514,
            1.60935891, 1.60150588, 1.59433031, 1.58781755, 1.58195436, 1.57673013,
            1.57213259, 1.56815553, 2.86658835, 4.17488813, 4.68163204, 7.41342402,
            7.40879059, 13.50699139, 13.45877552, 13.46566105, 13.52773762, 13.54493904,
            13.56733227, 10.67812157, 6.9712891, 7.97131395, 7.26442909, 6.27183151,
            5.78773785, 5.297194, 4.90160942, 4.55540562, 4.25638628, 3.99562502,
            3.9215436, 3.5702312, 3.52142835, 3.21958733, 3.07313967, 3.02650595,
            2.93608308, 2.8213985, 2.71632695, 2.61975694, 2.53074384, 2.44847775,
            2.37226176, 2.30149245, 2.23564649, 2.17426372, 2.11694193, 2.0633266,
            2.01310396, 1.96599472, 1.9217515, 1.88015223, 1.84099817, 1.804111,
            1.76933062, 1.73651075, 1.70552051, 1.67624104, 1.64856386, 1.62239015,
            1.64453673, 1.68725729, 1.73293197, 1.78184164, 1.83430707, 1.89069331,
            2.04831791, 2.1171186,  2.19155931, 2.27231956, 2.36018777, 1.51011825,
            1.4965167, 1.48371518, 1.47758424, 1.55210793, 3.10324192, 3.28008509,
            3.47970223, 3.70668197, 3.96694851, 4.26826477, 4.62101316, 5.03939867,
            5.54339314, 6.10359621, 1.43871272, 1.40497327, 1.39061165, 1.38594687]  

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
        self.assertEquals(min_laser, 1.38594687, "1.38594687!=1.38594687")
        self.assertFalse(collision, False)
        self.rc.shutdownhook()


if __name__ == '__main__':
    rosunit.unitrun(PKG, NAME, TestLibrary)