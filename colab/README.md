# Agent with Continuous Control

Training of 20 simultaneous agents, using Deep Deterministic Policy Gradients (DDPG).

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

## Environment
The observation consists of 33 variables of rotation, velocity and arm angular velocities. Each action is a vector with four numbers, corresponding to the torque applicable to two joints. Each entry in the action vector must be a number between -1 and 1.

### Goal
The task is episodic, and to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

### Reward
In this environment, a double-hinged arm can move to target locations. A +0.1 reward is provided for each step the agent's hand is at the objective location. So your agent's goal is to maintain your position at the target location for as many steps as possible.

## Starting

1. Clone this repository to your local machine using `git clone` .

```
https://github.com/Nicolasalan/Deep-Reinforcement-Learning-pytorch.git
```

2. Download the Unity ML environment from one of the links below based on your operating system:
    - Linux: [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32 bits): [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64 bits): [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


3. After downloading, extract the **zip** file and place the extracted folder in the root of the **Continuos-Control-rl** repository.

4. Change the unity environment's `file_name` path to match your operating system.
```
env = UnityEnvironment(file_name= "HERE")
```

5. Next, run all cells in the **continuos-control-rl.ipynb** notebook to train the agent.


