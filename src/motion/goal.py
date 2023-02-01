#!/usr/bin/env python3

from motion.srv import GoalPose

import rospy

class GoalPose():

     def __init__(self, CONFIG_PATH):
          rospy.Service('goal_pose', GoalPose, self.handler)
          self.CONFIG_PATH = CONFIG_PATH
          self.rate = rospy.Rate(5)
          
     def Goal(self, path):  

          # TODO: agent model training

          # TODO: agent model inference

          pass

     def handler(self, request):
          pass

if __name__ == '__main__':
    rospy.init_node('move_goal', log_level=rospy.INFO)
    GoalPose().Goal()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass