#!/usr/bin/env python3

from motion.srv import GoalPose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyResponse

import rospy
import yaml

class GoalPose():

     def __init__(self, CONFIG_PATH):
          rospy.Service('goal', GoalPose, self.handler)
          self.CONFIG_PATH = CONFIG_PATH
          self.param = self.useful.load_config("config.yaml")
          self.rate = rospy.Rate(5)

          self.state_dim = self.param["environment_dim"] + self.param["robot_dim"]
          self.action_dim = self.param["action_dim"]

          self.odom = rospy.Subscriber(self.param["topic_odom"], Odometry, queue_size=10)
          self.scan = rospy.Subscriber(self.param["topic_scan"], LaserScan, queue_size=10)

          self._check_odom_ready()
          self._check_scan_ready()

          rospy.loginfo("Finished waiting for the topics.")

     def _check_scan_ready(self):
          self.scan = None
          while self.scan is None and not rospy.is_shutdown():

               try:
                    self.scan = rospy.wait_for_message(self.param["topic_scan"], LaserScan, timeout=1.0)
                    rospy.logdebug("Current " + self.param["topic_scan"] + " READY=>" + str(self.scan))

               except:
                    rospy.logerr("Current " + self.param["topic_scan"] + " not ready yet, retrying.")

     def _check_odom_ready(self):
          self.odom = None
          while self.odom is None and not rospy.is_shutdown():
               try:
                    self.odom = rospy.wait_for_message(self.param["topic_odom"], Odometry, timeout=1.0)
                    rospy.logdebug("Current " + self.param["topic_odom"] + " READY=>" + str(self.odom))

               except:
                    rospy.logerr("Current " + self.param["topic_odom"] + " not ready yet, retrying.") 
     
     def Goal(self, x, y):  
          data = {'position': [x, y, 0]}
          with open(self.param["waypoints"], "w") as file:
               yaml.dump(data, file)

          rospy.loginfo("Position written to YAML: [{}, {}, 0]".format(x, y))  

     def handle_store_position(self, request):

          x = request.x
          y = request.y

          self.Goal(x, y)
          
          return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('move_goal', log_level=rospy.INFO)
    GoalPose()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass