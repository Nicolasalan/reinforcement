#!/usr/bin/env python3

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyResponse

import rospy
import yaml

class GoalPose():

     def __init__(self):
          rospy.Service('goal', GoalPose, self.handler)
          self.path = rospy.get_param("~path", "config/goals.yaml")
          self.rate = rospy.Rate(5)

          self._check_odom_ready()
          self._check_scan_ready()

          rospy.loginfo("Finished waiting for the topics.")
     
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