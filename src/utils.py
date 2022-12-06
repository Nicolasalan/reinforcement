def change_goal(self):
     # Place a new goal and check if its location is not on one of the obstacles
     if self.upper < 10:
          self.upper += 0.004
     if self.lower > -10:
          self.lower -= 0.004

     goal_ok = False

     while not goal_ok:
          self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
          self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
          goal_ok = check_pos(self.goal_x, self.goal_y)

def random_box(self):
     # Randomly change the location of the boxes in the environment on each reset to randomize the training
     # environment
     for i in range(4):
          name = "cardboard_box_" + str(i)

          x = 0
          y = 0
          box_ok = False
          while not box_ok:
               x = np.random.uniform(-6, 6)
               y = np.random.uniform(-6, 6)
               box_ok = check_pos(x, y)
               distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
               distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
               if distance_to_robot < 1.5 or distance_to_goal < 1.5:
               box_ok = False
          box_state = ModelState()
          box_state.model_name = name
          box_state.pose.position.x = x
          box_state.pose.position.y = y
          box_state.pose.position.z = 0.0
          box_state.pose.orientation.x = 0.0
          box_state.pose.orientation.y = 0.0
          box_state.pose.orientation.z = 0.0
          box_state.pose.orientation.w = 1.0
          self.set_state.publish(box_state)

def diff_angle():
     # Calculate the relative angle between the robots heading and heading toward the goal
     skew_x = self.goal_x - self.odom_x
     skew_y = self.goal_y - self.odom_y
     dot = skew_x * 1 + skew_y * 0
     mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
     mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
     beta = math.acos(dot / (mag1 * mag2))
     if skew_y < 0:
          if skew_x < 0:
               beta = -beta
          else:
               beta = 0 - beta
     theta = beta - angle
     if theta > np.pi:
          theta = np.pi - theta
          theta = -np.pi - theta
     if theta < -np.pi:
          theta = -np.pi - theta
          theta = np.pi - theta