[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif 
"Trained Agent"

[image2]: layers_96x88_585ep.png  "im2_96x88_585ep"
[image3]: layers_48x32_579ep.png  "im3_48x32_579ep"
[image4]: layers_80x88_572ep.png  "im4_80x88_572ep"
[image5]: layers_64x56_590ep.png  "im5_64x56_590ep"
[image6]: layers_80x88_633ep.png  "im6_80x88_633ep"

# Project 1: Navigation

### Introduction

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting 
a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while 
avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Completion criteria

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 
over 100 consecutive episodes.

### Getting Started

1. Follow the instructions in [lalopey/reinforcement-learning](https://github.com/lalopey/reinforcement-learning) to 
install all necessary packages and dependencies

2. Download the environment from one of the links below. For this project, you will  **not**  need to install Unity - this it's already been built. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

3. Place the file in the `unity_environments` folder in the current directory, and unzip (or decompress) the file. 


### Model

The problem is solved by using a Deep-Q Network. The specifics of the model and the results can be found in:

To run the model, run the [navigation.py](https://github.com/lalopey/reinforcement-learning/blob/master/1%20-%20Navigation%20-DQN/navigation.py)
file. Make sure you change the line:

**`env = UnityEnvironment(file_name="unity_environments/Banana_Linux/Banana.x86_64", worker_id=1)`**

to use the unity environment suited for your operating system.

A trained model can be found in the `trained_agents` directory. 


        