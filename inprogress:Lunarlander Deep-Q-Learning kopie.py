import numpy as np
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#setup the neural network
class neuralNetwork(nn.Module):
    def __init__(self, amount_of_inputs, amount_of_actions):
        super().__init__()
        #neural network consists of 2 fully connected linear layers
        self.fc1 = nn.Linear(in_features = amount_of_inputs, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 256)
        self.out = nn.Linear(in_features = 256, out_features = amount_of_actions)

    #forward function consists of 2 rectified linear function.
    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class lunarEnvironmentManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('LunarLander-v2')
        self.env.reset()
        self.done = False

    def select_action(self, state, neuralNet):
        self.env.reset()

    def reset(self):
        observation = self.env.reset()
        self.done = False
        return observation

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def amount_of_state_inputs(self):
        amount_state_inputs = len(self.env.observation_space.low)
        return amount_state_inputs

    def amount_of_actions(self):
        amount_of_actions = self.env.action_space.n
        return amount_of_actions

    def take_action_and_observe(self, action):
        observation_, reward, self.done, info = self.env.step(action)
        return observation_, reward, self.done

class lunarLanderAgent():
    def __init__(self, num_actions, device, startEpsilon, endEpsilon, epsilonDecay):
        self.current_step   = 0
        self.num_actions    = num_actions
        self.device         = device
        self.startEpsilon   = startEpsilon
        self.endEpsilon     = endEpsilon
        self.epsilonDecay   = epsilonDecay

    # determine the exploration rate, which is used for exploration or exploitation, this is done epsilon greedy
    def determine_exploration_rate(self, start, end, decay):
        exploration_rate = end + (start - end) * math.exp(-1. * self.current_step * decay)
        self.current_step += 1
        return exploration_rate

    def select_action(self, state, policy_network):
        rate = self.determine_exploration_rate(self.startEpsilon, self.endEpsilon, self.epsilonDecay)

        if rate > random.random(): #explore by taking random action
            action = random.randrange(self.num_actions)
            return action

        with torch.no_grad(): #else do an action fed through the policy network, and return the action with highest q-value!
            return policy_network.forward(state).argmax().item()

class replayMemory():
    def __init__(self, memory_capacity, device, input_dims,batch_size):
        self.memory_capacity = memory_capacity
        self.memory = []
        self.device = device
        self.amountOfExperiencesPushed = 0
        self.batch_indeces      = np.empty(batch_size)
        self.state_memory       = np.zeros((self.memory_capacity, input_dims), dtype= np.float32)
        self.new_state_memory   = np.zeros((self.memory_capacity, input_dims), dtype= np.float32)
        self.reward_memory      = np.zeros(self.memory_capacity, dtype= np.float32)
        self.terminal_memory    = np.zeros(self.memory_capacity, dtype= bool)
        self.action_memory      = np.zeros(self.memory_capacity, dtype= np.int32)

    #push the experience of the timestep into the experience memory
    def pushExperienceInMemory(self, state, new_state, reward, action, done):
        # if the memory is not yet full, push the experience into the memory, else push it into the beginning of the memory
        index = self.amountOfExperiencesPushed % self.memory_capacity
        self.state_memory[index]        = state
        self.reward_memory[index]       = reward
        self.new_state_memory[index]    = new_state
        self.action_memory[index]       = action
        self.terminal_memory[index]     = done

        self.amountOfExperiencesPushed += 1

    def canMemoryProvideSampleOfBatchSize(self, batch_size):
        return self.amountOfExperiencesPushed >= batch_size

    #the samples are stored in numpy arrays and therefore need to be converted to tensors
    def takeSampleFromMemoryAndConvertToTensor(self, batch_size):
        amount_experiences_in_memory = min(self.amountOfExperiencesPushed, self.memory_capacity)
        self.batch_indeces = np.random.choice(amount_experiences_in_memory, batch_size) #select random indeces from the memory to sample

        #Take random indeces, extract them from the memory and put them into tensors, so they can be used by the neural network
        state_tensor     = torch.tensor(self.state_memory[self.batch_indeces]).to(self.device)
        reward_tensor    = torch.tensor(self.reward_memory[self.batch_indeces]).to(self.device)
        new_state_tensor = torch.tensor(self.new_state_memory[self.batch_indeces]).to(self.device)
        terminal_tensor  = torch.tensor(self.terminal_memory[self.batch_indeces]).to(self.device)

        action_array     = self.action_memory[self.batch_indeces]

        return state_tensor, reward_tensor, new_state_tensor, action_array, terminal_tensor

    def learnFromMemory(self, policy_network, target_network, batch_size, optimizer, gamma):
        #Get random samples to learn from
        state_tensor, reward_tensor, new_state_tensor, action_array, terminal_tensor = self.takeSampleFromMemoryAndConvertToTensor(batch_size)
        batch_index = np.arange(batch_size, dtype=np.int32) #make a numpy array from 0 to number of batch size. Needed for the forward pass through the policy_network

        #pass the q_values through the policy network to get the current q-value and through the target network in order to get the next q-value
        q_values_current_state  = policy_network.forward(state_tensor)[batch_index, action_array]
        q_values_next_state     = target_network.forward(new_state_tensor)
        q_values_next_state[terminal_tensor] = 0.0 #if the next q-value represents a terminal state we do not want to learn from this, hence we set the q-value to 0

        q_values_next_state = (reward_tensor * gamma) + torch.max(q_values_next_state, dim=1)[0]

        #Determine the loss and do backpropagation on the policy network
        lossFunction = F.mse_loss(q_values_current_state, q_values_next_state)
        optimizer.zero_grad()
        lossFunction.backward()
        optimizer.step()

def plot(values, reward_avg):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    plt.plot(reward_avg)
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)

#Determine constants for the agent
BATCH_SIZE          = 50           #Batches we want to update our policy network from
GAMMA               = 0.99         #The discount factor for every step that is taken.
EPSILON_START       = 1            #The epsilon value we want to start with indicating exploration
EPSILON_END         = 0.01         #The value for epsilon when we want to stop explore
EPSILON_DECAY       = 0.001        #How fast will epsilon_decay?
TARGET_UPDATE       = 6            #The amount of episodes after which we want to
MEMORY_SIZE         = 100000       #Capacity for the memory and thus the amount we want to store
AMOUNT_OF_EPISODES  = 1000         #The amount of episodes we want to play the game for training the network
LEARNING_RATE       = 0.001

#Setup the device, the environmentManager and the agent
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environmentManager      = lunarEnvironmentManager(device)
agent                   = lunarLanderAgent(environmentManager.amount_of_actions()\
                                           , device, EPSILON_START, EPSILON_END, EPSILON_DECAY)

#make a target and policy network. We copy all weights and biases of the policy network to the target network.
policy_network          = neuralNetwork(environmentManager.amount_of_state_inputs(), environmentManager.amount_of_actions()).to(device)
target_network          = neuralNetwork(environmentManager.amount_of_state_inputs(), environmentManager.amount_of_actions()).to(device)
target_network.load_state_dict(policy_network.state_dict()) #Copy weights and biases of policy network to target network
target_network.eval()
episodes_for_target_network_update  = 30

#Initialize the replayMemory and the optimizer. In this case we use Adam as an optimizer. There is no real reason to use Adam, just because it's the standard
replayMemory            = replayMemory(MEMORY_SIZE,device,environmentManager.amount_of_state_inputs(), BATCH_SIZE)
optimizer               = optim.Adam(params=policy_network.parameters(), lr=LEARNING_RATE)

#Initialize some variables for plotting the progress of the agent
total_reward_for_episodes   = []
reward_average              = [0]

for episode in range(AMOUNT_OF_EPISODES):
    total_reward_for_episode    = 0
    state                       = torch.tensor(environmentManager.reset())
    print(episode)
    while not environmentManager.done:
        action                        = agent.select_action(state, policy_network) #select an action according to highest q-value retrieved from the policy network
        new_state, reward, done       = environmentManager.take_action_and_observe(action)
        new_state = torch.tensor(new_state)
        replayMemory.pushExperienceInMemory(state, new_state, reward, action, done)
        state = new_state
        total_reward_for_episode += reward

        if episode > (AMOUNT_OF_EPISODES-10):
            environmentManager.render()

        if replayMemory.canMemoryProvideSampleOfBatchSize(BATCH_SIZE):
            replayMemory.learnFromMemory(policy_network, target_network, BATCH_SIZE, optimizer, GAMMA)

    print("Reward for episode")
    print(total_reward_for_episode)
    target_network_update_needed = episode % episodes_for_target_network_update
    if target_network_update_needed == 0:
        print("updating target network")
        target_network.load_state_dict(policy_network.state_dict())
        target_network.eval()

    total_reward_for_episodes.append(total_reward_for_episode)
    reward_average.append(sum(total_reward_for_episodes)/(len(total_reward_for_episodes)))

plot(total_reward_for_episodes, reward_average)
environmentManager.close()