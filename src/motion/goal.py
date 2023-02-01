#!/usr/bin/env python3

from motion.srv import GoalPose

import rospy
import torch
import numpy as np

from agent import Agent
from environment import Env
from utils import Extension

class GoalPose():

     def __init__(self, CONFIG_PATH):
          rospy.Service('goal_pose', GoalPose, self.handler)
          self.CONFIG_PATH = CONFIG_PATH
          self.param = self.useful.load_config("config.yaml")
          self.rate = rospy.Rate(5)

          self.state_dim = self.param["environment_dim"] + self.param["robot_dim"]
          self.action_dim = self.param["action_dim"]

          
     def Goal(self):  

          # TODO: agent model training
          agent = Agent(state_size=self.state_dim, action_size=self.action_dim, random_seed=42, CONFIG_PATH=self.CONFIG_PATH)
          env = Env(self.CONFIG_PATH)

          agent.actor_local.load_state_dict(torch.load(self.param["model_actor"]))
          agent.critic_local.load_state_dict(torch.load(self.param["model_critic"]))  
           
          done = False

          agent.reset()                                               # reset environment    
          states = env.reset_env()                                    # get the current state of each agent

          max_t = 1000

          for t in range(max_t):   
                         
               action = agent.action(states)                          # choose an action for each agent
               actions = [(action[0] + 1) / 2, action[1]]

               next_states, rewards, done, _ = env.step_env(actions)  # send all actions to the environment

               # save the experiment in the replay buffer, run the learning step at a defined interval
               agent.step(states, actions, rewards, next_states, done, t)

               states = next_states
               if np.any(done):                                       # exit loop when episode ends
                    break

     def handler(self, request):
          pass

if __name__ == '__main__':
    rospy.init_node('move_goal', log_level=rospy.INFO)
    GoalPose().Goal()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass