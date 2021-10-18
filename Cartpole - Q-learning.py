import lr as lr
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time, math, random
from typing import Tuple

import gym

env = gym.make("CartPole-v0")

#initialization
amount_of_episodes = 1000
amount_of_steps = 10
discount_factor = 1
min_learning_rate = 0.1
min_exploration_rate = 0.1

#make q_table
print(env.observation_space)
observation_space = env.observation_space
action_space = env.action_space.n

bins = (6, 12) #What do these bins represent?

lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]
q_table = np.zeros(bins + (env.action_space.n,))

rene_en_daan = env.
reneEnDaan =


#discretizing function #How does this discretization work?
def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

def policy(state):
    optimal_policy = np.argmax(q_table[state]) #Chose the best action; left or right. either action has a q-value
    return optimal_policy

def new_q_value(reward, new_state, discount_factor):
    future_optimal_value = np.max(q_table[new_state])
    learned_value = reward + (discount_factor * future_optimal_value)
    return learned_value

# Adaptive learning of Learning Rate
def new_learning_rate(episodes, min_learning_rate): #n is the amount of episodes, so...the learning rate decays with the amount of amount_of_episodes
    generated_learning_rate_from_episodes = min(1.0, 1.0 - math.log10((episodes + 1) / 25)) # it is either 1 or a lower generated number
    decayed_learning_rate = max(min_learning_rate, generated_learning_rate_from_episodes) #if the generated learning rate becomes smaller than min, select min
    return decayed_learning_rate

def new_exploration_rate(episodes, min_exploration_rate):
    generated_exploration_rate_from_episodes = min(1, 1.0 - math.log10((episodes  + 1) / 25))
    decayed_exploration_rate = max(min_exploration_rate, generated_exploration_rate_from_episodes)
    return decayed_exploration_rate

for e in range(amount_of_episodes):

    # discretize state into buckets
    current_state, done = discretizer(*env.reset()), False #????? -> Question to Tonio

    while not done:

        #generate a random exploration rate and explore when its bigger than the exploration rest, else exploit the env.
        if np.random.random() > new_exploration_rate(e, min_exploration_rate):
            action = env.action_space.sample()  # explore
        else :
            action = policy(current_state)  # exploit
        # increment enviroment
        obs, reward, done, info = env.step(action)
        new_state = discretizer(*obs)

        # Update Q-Table
        learning_rate = new_learning_rate(e, min_learning_rate)
        learnt_value = new_q_value(reward, new_state, discount_factor)
        old_value = q_table[current_state][action]
        q_table[current_state][action] = (1 - learning_rate) * old_value + learning_rate * learnt_value

        current_state = new_state
        print(e)
        # Render the cartpole environment
        env.render()
print("done learning")
#env.close()

#for e in range(3):
 #   env.reset()

  #  while ot

#replace q-table with functional approximator
#------

# make network 1
# make network 2

#wire together network


"""
#initialization
amount_of_episodes = 3
amount_of_steps = 10

def policy(observation):
    return 1

for episodes in range(amount_of_episodes):
    observation = env.reset()

    for steps in range(amount_of_steps):
        action = policy(observation)
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)
env.close()
"""