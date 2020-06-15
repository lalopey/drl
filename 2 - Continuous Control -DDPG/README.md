[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The version used for this project contains a single agent.

#### Completion Criteria

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Follow the instructions in [lalopey/drl](https://github.com/lalopey/drl) to 
install all necessary packages and dependencies

2. Download the environment from one of the links below. For this project, you will  **not**  need to install Unity - this it's already been built. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

3. Place the file in the `unity_environments` folder in the current directory, and unzip (or decompress) the file. 

### Model

The problem is solved by using a Deep Deterministic Policy Gradient (DDPG) architecture. 

- The code for the agents can be found in [drltools/agent.py](https://github.com/lalopey/drl/blob/master/drltools/agent/agent.py)
- The code for the PyTorch models can be found in [drltools/model.py](https://github.com/lalopey/drl/blob/master/drltools/model/model.py)

To run the model for a single agent, run the [continuous_control_ddpg_reacher.py](https://github.com/lalopey/drl/blob/master/2%20-%20Continuous%20Control%20-DDPG/continuous_control_ddpg_reacher.py)
file. Make sure you change the line:

**`env = UnityEnvironment(file_name="unity_environments/Reacher_Linux/Banana.x86_64", worker_id=1)`**

to use the unity environment suited for your operating system.

For the model with 20 agents, run To run the model for a single agent, run the [continuous_control_ddpg_reacher20.py](https://github.com/lalopey/drl/blob/master/2%20-%20Continuous%20Control%20-DDPG/continuous_control_ddpg_reacher20.py)

A trained model for each environment can be found in the `trained_agents` directory. 