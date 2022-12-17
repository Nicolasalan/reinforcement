#! /usr/bin/env python3

from motion.utils import Extension
import unittest
import rosunit

class CaseA(unittest.TestCase):

    """
    Test: angulo do robo em relacao ao alvo
    ======
        Input (float): odom x, odom y, goal x, goal y, yaw
        Output (float): theta
    """
    def setUp(self):
        self.rc = Extension()

    def test_angles(self):

        resp = self.angles()
        self.assertEquals()
        self.rc.shutdownhook()

class CaseB(unittest.TestCase):
    """
    Test: distancia entre o robo e o alvo
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """
    def setUp(self):
        self.rc = Extension()

    def test_distance_to_goal(self):

        resp = self.rc.distance_to_goal()
        self.assertTrue()
        self.rc.shutdownhook()

class CaseC(unittest.TestCase):
    """
    Test: lista de posicoes para randomizar
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """ 
    def setUp(self):
        self.rc = Extension()

    def test_path_goal(self):

        resp = self.path_goal()
        self.assertEquals()
        self.rc.shutdownhook()

class CaseD(unittest.TestCase):
    """
    Test: randomizacao das posicoes x e y
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """ 
    def setUp(self):
        self.rc = Extension()

    def test_random_goal(self):

        resp = self.rc.random_goal()
        self.assertTrue()
        self.rc.shutdownhook()

class CaseE(unittest.TestCase):
    """
    Test: recompensa do agent
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """
    def setUp(self):
        self.rc = Extension()

    def test_get_reward(self):

        resp = self.rc.get_reward()
        self.assertTrue()
        self.rc.shutdownhook()

class CaseF(unittest.TestCase):
    """
    Test: verificacao de colisao
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """ 
    def setUp(self):
        self.rc = Extension()

    def test_observe_collision(self):

        resp = self.rc.observe_collision()
        self.assertTrue()
        self.rc.shutdownhook()

class CaseG(unittest.TestCase):
    """
    Test: randomizacao do alvo
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """ 
    def setUp(self):
        self.rc = Extension()

    def test_change_goal(self):

        resp = self.rc.change_goal()
        self.assertTrue()
        self.rc.shutdownhook()

class CaseH(unittest.TestCase):
    """
    Test: verificar posicao em duplicidade
    ======
        Input (int): número máximo de episódios de treinamento
        Output (int): número máximo de timesteps por episódio
    """ 
    def setUp(self):
        self.rc = Extension()

    def test_check_pose(self):

        resp = self.rc.check_pose()
        self.assertTrue()
        self.rc.shutdownhook()

class TestSuite(unittest.TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(CaseA())
        self.addTest(CaseB())
        self.addTest(CaseC())
        self.addTest(CaseD())
        self.addTest(CaseE())
        self.addTest(CaseF())
        self.addTest(CaseG())
        self.addTest(CaseH())

# rosunit
rosunit.unitrun('motion', 'integration_test',
                'integration_test.MyTestSuite')