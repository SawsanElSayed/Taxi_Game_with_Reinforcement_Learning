
## **Reinforcement Learning Project**
***
![image](https://www.learndatasci.com/documents/14/Reinforcement-Learning-Animation.gif)

**Table of Contents**




#### Overview
In this assignment, we have 3 environments which you can choose from them:
    1-CliffWalking-v0
    2-FrozenLake-v1
    3-Taxi-v3
The default environment is Taxi Problem definition which there are 4 locations assigned by different letters.
Our mission is to pick up the passenger at one of these locations then drop him off at another one. We receive +20 points for a successful drop-off and lose 1 point for every time-step it takes. There is also a 10 points penalty for illegal pick-up and drop-off actions. Introduction In this assignment we'll implement the Q-leaning algorithm then we will see how the decay of the hyperparameter such as learning rate and discount factor and epsilon  will affect the results. We generalize this code to can be suitable for any environment.
#### Requirements
You should setup this library to run the code:
```python
!pip install gym
```
##### Taxi-v3 environment info (Default Environment)

![image](https://drive.google.com/uc?export=view&id=1WOvwLRDazOzUfsyEYpB6QZXejppLfRPQ)

The mission is to pick up the passenger at one of these locations then drop him off at another one. There are a few points which we want the taxi do it:
- Drop off the passenger in the right location.
- Save passenger's time by taking shortest way and minimum time possible to drop off.
- Take care of passenger's safety and traffic rules.

#### Coding
##### Step(1): Importing the libraries:
```python
import gym 
from IPython.display import clear_output
from time import sleep
import random
from IPython.display import clear_output
import numpy as np
```
##### Step(2): Set up the environment:
```python
def agent_enviro (state):
  env = gym.make("Taxi-v3").env
  env.render()
  env.reset() # reset environment to a new, random state
  env.s = state
  env.render()
  print("Action Space {}".format(env.action_space))
  print("State Space {}".format(env.observation_space))
  return state,env
```
```python
state,env = agent_enviro(414)
state
```

![image](https://drive.google.com/uc?export=view&id=1rqcW75H1NNK6aa1mZ3dw6wrv9iLVOwTX)

##### Step(3): Training the agent
**Q-learning approach using decaying hyper parameters while training:**
```python
def Train_the_agent(game, alpha,gamma,epsilon):
  %%time
  """Training the agent"""
  env = gym.make(game).env
  env.reset()
  env.render()
  decay_factor=1e-3

  # Initialize the q table
  q_table = np.zeros([env.observation_space.n, env.action_space.n])
  # Initialize the q table
  q_table = np.zeros([env.observation_space.n, env.action_space.n])

  # Hyperparameters
  #alpha = 0.1
  #gamma = 0.6
  #epsilon = 0.1

  # For plotting metrics
  all_epochs = []
  all_penalties = []

  for i in range(1, 100001):
      state = env.reset()

      epochs, penalties, reward, = 0, 0, 0
      done = False
      if i % 10000==0:
        epsilon-=decay_factor
      
      while not done:
          if random.uniform(0, 1) < epsilon:
              action = env.action_space.sample() # Explore action space
          else:
              action = np.argmax(q_table[state]) # Exploit learned values

          next_state, reward, done, info = env.step(action) 
          
          old_value = q_table[state, action]
          next_max = np.max(q_table[next_state])
      
          new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
          q_table[state, action] = new_value

          if reward == -10:
              penalties += 1

          state = next_state
          epochs += 1
          
      if i % 100 == 0:
          clear_output(wait=True)
          print(f"Episode: {i}")

  print("Training finished.\n")
  return q_table
```
```python
q_table = Train_the_agent("Taxi-v3",0.1,0.6,0.1)
print(q_table)
```

![image](https://drive.google.com/uc?export=view&id=1KLRFy4hg0fcnWx7c4iMA_QlIONQX3j40)

##### Step(4): Evaluation
Evaluate agent's performance after Q-learning
```python
def Evaluation(q_table):
  total_epochs, total_penalties = 0, 0
  episodes = 1000

  for _ in range(episodes):
      state = env.reset()
      epochs, penalties, reward = 0, 0, 0
      
      done = False
      
      while not done:
          action = np.argmax(q_table[state])
          state, reward, done, info = env.step(action)

          if reward == -10:
              penalties += 1

          epochs += 1

      total_penalties += penalties
      total_epochs += epochs

   print(f"Results after {episodes} episodes:")
   print(f"Average timesteps per episode: {total_epochs / episodes}")
   print(f"Average penalties per episode: {total_penalties / episodes}")
```
```python
Evaluation(q_table)
```
![image](https://drive.google.com/uc?export=view&id=1D6BZ_8AEL0R_FtwXiMiVz4XOE9h8EMLR)
