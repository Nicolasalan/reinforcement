#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from squaternion import Quaternion

from utils import Extension

class ContinuousEnv():
     def __init__(self, CONFIG_PATH):

          self.useful = Extension(CONFIG_PATH)
          rospy.init_node("gym", anonymous=True)

          # Function to load yaml configuration file
          param = self.useful.load_config("config.yaml")

          # set the initial state
          self.goal_reached_dist = param["GOAL_REACHED_DIST"]
          self.environment_dim = param["ENVIRONMENT_DIM"]
          self.time_delta = param["TIME_DELTA"]
          self.collision_dist = param["COLLISION_DIST"]
          self.robot = param["ROBOT"]
          self.orientation_threshold = param["ORIENTATION_THRESHOLD"]
          self.noise_sigma = param["NOISE_SIGMA"]
          self.cmd = param["TOPIC_CMD"]
          self.odom = param["TOPIC_ODOM"]
          self.scan = param["TOPIC_SCAN"]

          # initialize global variables
          self.odom_x = 0.0
          self.odom_y = 0.0
          self.goal_x = 0.0
          self.goal_y = 0.0

          self.scan_data = np.ones(self.environment_dim) * 10
          self.path_targets = param["path_goal"] + '/goal.yaml'
          self.goals = self.useful.poses(self.path_targets)
          self.last_odom = None

          # ROS publications and subscriptions
          self.pub_cmd_vel = rospy.Publisher(self.cmd, Twist, queue_size=10)
          self.odom = rospy.Subscriber(self.odom, Odometry, self.odom_callback, queue_size=10)
          self.scan = rospy.Subscriber(self.scan, LaserScan, self.scan_callback)

          self.gaps = self.useful.array_gaps(self.environment_dim)
          self.count_goals = 0
          self.idx = 0
          rospy.sleep(1)

     def odom_callback(self, msg):
          """
          Processes an odometry message and updates the current pose of the robot.
          ======
          Params:
               odom_msg (Odometry): An odometry message containing the current pose of the robot.
               pose (Pose): The current pose of the robot, represented as a Pose object.
          """
          self.last_odom = msg.pose.pose

     def scan_callback(self, scan):
          """
          Sensor scan message that contains range data from a laser range finder
          ======
          Params:
               scan (LaserScan): list of range measurements, one for each beam in the scan.
               scan_data (array): A list of range measurements.
          """
          data = scan.ranges
          self.scan_data = self.useful.scan_rang(self.environment_dim, self.gaps, data)

     def step_env(self, action):
          """
          Traverse the environment in a robot simulation
          ======
          Params:
               action (array): linear and angular speeds based on action values
               state (array): array with robot states (leisure, speed, distance and theta)
               reward (gloat): agent reward function
               done (bool): check if it collided with an object
               target (bool): check if you reached the target
          """

          rospy.logwarn('Step Environment             => Stepping environment ...')
          target = False

          # ================== PUBLISH ACTION ================== #

          try:
               # Publish the robot action
               vel_cmd = Twist()
               vel_cmd.linear.x = action[0] * self.cmd_linear
               vel_cmd.angular.z = action[1] * self.cmd_angular
               self.pub_cmd_vel.publish(vel_cmd)
               rospy.loginfo('Publish Action               => Linear: ' + str(vel_cmd.linear.x) + ' Angular: ' + str(vel_cmd.angular.z))

          except:
               rospy.logerr('Publish Action              => Failed to publish action')

          # ================== READ SCAN DATA ================== #
          
          done, collision, min_laser = self.useful.observe_collision(self.scan_data, self.collision_dist)

          try:
               v_state = []
               v_state[:] = self.scan_data[:]

               # add noise to the laser data
               noisy_state = np.clip(v_state + np.random.normal(0, self.noise_sigma, len(v_state)), 0, 10.0)
               state_laser = [list(noisy_state)]

               rospy.loginfo('Read Scan Data               => Min Lazer: ' + str(min_laser) + ' Collision: ' + str(collision) + ' Done: ' + str(done))
          
          except:
               rospy.logerr('Read Scan Data              => Error reading scan data')
               done = False
               collision = False
               min_laser = 0.0
               state_laser = [np.ones(self.environment_dim) * 10]

          # ================== READ ODOM DATA ================== #
          try:
               self.odom_x = self.last_odom.position.x
               self.odom_y = self.last_odom.position.y

               quaternion = Quaternion(
                    self.last_odom.orientation.x,
                    self.last_odom.orientation.y,
                    self.last_odom.orientation.z,
                    self.last_odom.orientation.w
               )
               euler = quaternion.to_euler(degrees=False)
               angle = round(euler[2], 4)

               rospy.loginfo('Read Odom Data               => Odom x: ' + str(self.odom_x) + ' Odom y: ' + str(self.odom_y) + ' Angle: ' + str(angle))

          except:
               rospy.logfatal('Read Odom Data              => Error reading odometry data')
               self.odom_x = 0.0
               self.odom_y = 0.0
               angle = 0.0
     
          # ================== CALCULATE DISTANCE AND THETA ================== #
          # Calculate distance to the goal from the robot
          distance = self.useful.distance_to_goal(self.odom_x, self.goal_x, self.odom_y, self.goal_y)
          # Calculate the relative angle between the robots heading and heading toward the goal
          theta = self.useful.angles(self.odom_x, self.goal_x, self.odom_y, self.goal_y, angle)

          rospy.loginfo('Calculate distance and angle => Distance: ' + str(distance) + ' Angle: ' + str(theta))

          # ================== ORIENTATION GOAL ================== #
          # Calculate difference between current orientation and target orientation
          orientation_diff = abs(angle - self.goal_orientation)

          rospy.loginfo('Orientation Goal             => Orientation Diff: ' + str(orientation_diff))
          
          # ================== CALCULATE DISTANCE AND ANGLE ================== #
          # Detect if the goal has been reached and give a large positive reward
          if distance < self.goal_reached_dist: #and orientation_diff < self.orientation_threshold:
               target = True
               done = True
          
          rospy.loginfo('Check (Collided or Arrive)   => Target: ' + str(target) + ' Done: ' + str(done))

          # ================== SET STATE ================== #
          robot_state = [distance, theta, action[0], action[1]]
          state = np.append(state_laser, robot_state)
          reward = self.useful.get_reward(target, collision, action, min_laser)

          rospy.loginfo('Get Reward                   => Reward: ' + str(reward))
          return state, reward, done, target

     def reset_env(self):
          """
          Resets the environment to a new position and sets the initial robot states
          ======
          Params:
               state (array): array with robot states (leisure, speed, distance and theta)
          """

          # ================== SET ANGLE ================== #

          try:
               self.odom_x = self.last_odom.position.x
               self.odom_y = self.last_odom.position.y

               quaternion = Quaternion(
                    self.last_odom.orientation.x,
                    self.last_odom.orientation.y,
                    self.last_odom.orientation.z,
                    self.last_odom.orientation.w
               )
               euler = quaternion.to_euler(degrees=False)
               angle = round(euler[2], 4)

               rospy.loginfo('Set Angle Robot              => Angle: ' + str(angle))
          
          except:
               rospy.logerr('Set Angle Robot           => Error setting random angle')
               angle = 0.0

          path = self.goals

          self.goal_x, self.goal_y, self.goal_orientation = path[self.idx]

          # ================== SET ORIENTATION ================== #
          try:

               rospy.loginfo('Set Angle Target             => Angle: ' + str(self.goal_orientation))
          
          except:
               rospy.logerr('Set Orientation                => Error setting random orientation')
               self.goal_orientation = 0.0

          # ================== SET POSITION ================== #
    
          rospy.loginfo('Set Position                 => Goal: (' + str(self.goal_x) + ', ' + str(self.goal_y) + ') Robot: (' + str(self.odom_x) + ', ' + str(self.odom_y) + ')')

          # ================== GET STATE SCAN ================== #
          try:
               v_state = []
               v_state[:] = self.scan_data[:]
               laser_state = [v_state]

               rospy.loginfo('Get state scan               => Laser: ' + str(np.mean(laser_state)))
          except:
               rospy.logerr('Get state scan              => Error getting state scan')

          # ==================CALCULATE DISTANCE AND ANGLE ================== #
          # Calculate distance to the goal from the robot
          distance = self.useful.distance_to_goal(self.odom_x, self.goal_x, self.odom_y, self.goal_y)
          # Calculate the relative angle between the robots heading and heading toward the goal
          theta = self.useful.angles(self.odom_x, self.goal_x, self.odom_y, self.goal_y, angle)

          rospy.loginfo('Calculate distance and angle => Distance: ' + str(distance) + ' Angle: ' + str(theta))
          print('========================================================================================================================')

          # ================== CREATE STATE ARRAY ================== #
          robot_state = [distance, theta, 0.0, 0.0]
          state = np.append(laser_state, robot_state)

          self.idx += 1

          # ================== RETURN STATE ================== #
          return np.array(state)
     
     def count(self):
          """
          Counts the number of goals reached
          """
          self.count_goals += 1
          rospy.loginfo('Count Goals Reached          => Goals: ' + str(self.count_goals))