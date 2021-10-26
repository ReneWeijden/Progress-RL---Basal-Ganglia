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
    def __init__(self, amount_of_inputs, amount_of_actions, dueling_q_learning, double_dueling_q_learning):
        self.dueling_q_learning        = dueling_q_learning
        self.double_dueling_q_learning = double_dueling_q_learning
        super().__init__()
        if self.dueling_q_learning or double_dueling_q_learning:
            self.fc1 = nn.Linear(in_features = amount_of_inputs, out_features = 128)
            self.fc2 = nn.Linear(in_features = 128, out_features = 128)
            self.V   = nn.Linear(in_features = 128, out_features = 1) # The value state function only outputs 1 parameter
            self.A   = nn.Linear(in_features = 128, out_features = amount_of_actions)
        else:
        #neural network consists of 2 fully connected linear layers
            self.fc1 = nn.Linear(in_features = amount_of_inputs, out_features = 256)
            self.fc2 = nn.Linear(in_features = 256, out_features = 256)
            self.out = nn.Linear(in_features = 256, out_features = amount_of_actions)

    #forward function consists of 2 rectified linear function.
    def forward(self, t):
        if self.dueling_q_learning or double_dueling_q_learning:
            layer_one = F.relu(self.fc1(t))
            layer_two = F.relu(self.fc2(layer_one))
            V         = self.V(layer_two) #The layers of the value function and the advantage function are not
            A         = self.A(layer_two)

            return V, A
        else:
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

    def select_action(self, state, policy_network, dueling_q_learning, double_dueling_q_learning):
        rate = self.determine_exploration_rate(self.startEpsilon, self.endEpsilon, self.epsilonDecay)

        if rate > random.random(): #explore by taking random action
            action = random.randrange(self.num_actions)
            return action

        if dueling_q_learning or double_dueling_q_learning:
            _, advantages_of_actions = policy_network.forward(state)
            most_advantageous_action = advantages_of_actions.argmax().item()

            return most_advantageous_action

        #by default use the standard forward through a standard DQN
        with torch.no_grad(): #else do an action fed through the policy network, and return the action with highest q-value!
            return policy_network.forward(state).argmax().item()

class replayMemory():
    def __init__(self, memory_capacity, device, input_dims,batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.device = device
        self.amountOfExperiencesPushed = 0
        self.probability_alpha  = 0.6
        self.probability_beta   = -0.4
        self.batch_indeces      = np.empty(batch_size)
        self.state_memory       = np.zeros((self.memory_capacity, input_dims), dtype= np.float32)
        self.new_state_memory   = np.zeros((self.memory_capacity, input_dims), dtype= np.float32)
        self.reward_memory      = np.zeros(self.memory_capacity, dtype= np.float32)
        self.terminal_memory    = np.zeros(self.memory_capacity, dtype= bool)
        self.action_memory      = np.zeros(self.memory_capacity, dtype= np.int32)
        self.priority_memory    = np.zeros(self.memory_capacity, dtype= np.float32)

    #push the experience of the timestep into the experience memory
    def pushExperienceInMemory(self, state, new_state, reward, action, done):
        # if the memory is not yet full, push the experience into the memory, else push it into the beginning of the memory
        max_prio = self.priority_memory.max() if self.memory else 1.0
        index = self.amountOfExperiencesPushed % self.memory_capacity
        self.state_memory[index]        = state
        self.reward_memory[index]       = reward
        self.new_state_memory[index]    = new_state
        self.action_memory[index]       = action
        self.terminal_memory[index]     = done

        self.priority_memory[index] = max_prio

        self.amountOfExperiencesPushed += 1

    def canMemoryProvideSampleOfBatchSize(self, batch_size):
        return self.amountOfExperiencesPushed >= batch_size

    #the samples are stored in numpy arrays and therefore need to be converted to tensors
    def takeSampleFromMemoryAndConvertToTensor(self, batch_size):
        self.amount_experiences_in_memory = min(self.amountOfExperiencesPushed, self.memory_capacity)
        prios = self.priority_memory[:self.amount_experiences_in_memory]
        probs = prios ** self.probability_alpha
        probs /= probs.sum()
        # select indeces from the memory to sample with the priority_memory as indicator for which indexes should be prioritized
        self.batch_indeces = np.random.choice(self.amount_experiences_in_memory, batch_size, p=probs)

        #Take random indeces, extract them from the memory and put them into tensors, so they can be used by the neural network
        state_tensor     = torch.tensor(self.state_memory[self.batch_indeces]).to(self.device)
        reward_tensor    = torch.tensor(self.reward_memory[self.batch_indeces]).to(self.device)
        new_state_tensor = torch.tensor(self.new_state_memory[self.batch_indeces]).to(self.device)
        terminal_tensor  = torch.tensor(self.terminal_memory[self.batch_indeces]).to(self.device)

        action_array     = self.action_memory[self.batch_indeces]
        weights          = self.calculateWeights(self.priority_memory, self.probability_alpha, self.probability_beta, self.batch_indeces)

        return state_tensor, reward_tensor, new_state_tensor, action_array, terminal_tensor, weights, self.batch_indeces

    def calculateWeights(self, priority_array, probability_alpha, probability_beta, batch_indeces):
        priorities             = priority_array

        probability_array      = pow(priorities, probability_alpha) / pow(priorities.sum(), probability_alpha)

        one_over_buffer_length = 1 / self.amount_experiences_in_memory
        one_over_prob_array    = 1 / probability_array
        weights                = pow((one_over_buffer_length * one_over_prob_array), probability_beta)

        normalized_weights = weights/max(weights)

        return normalized_weights[batch_indeces]

    def updatePriorities(self, batch_indeces, new_priorities):
        print("new priorities are")
        print(new_priorities)
        print("batchindeces")
        print(batch_indeces)
        for idx, prio in zip(batch_indeces, new_priorities):
            self.priority_memory[idx] = prio
            print(self.priority_memory[idx])
            print(self.priority_memory)
        self.priority_memory /= self.priority_memory.max() #normalize the distribution

    def learnFromMemory(self, device, policy_network, target_network, batch_size, optimizer, gamma, double_q_learning, deep_q_learning, dueling_q_learning, double_dueling_q_learning):
        #Get random samples to learn from
        state_tensor, reward_tensor, new_state_tensor, action_array, terminal_tensor, weights, sample_indices \
            = self.takeSampleFromMemoryAndConvertToTensor(batch_size)
        batch_index = np.arange(batch_size, dtype=np.int32) #make a numpy array from 0 to number of batch size. Needed for the forward pass through the policy_network
        action_tensor = torch.tensor(action_array, dtype=torch.int64).to(device) # convert action array to action tensor for dueling networks

        if dueling_q_learning:
            #retrieve the state value V and advantage value A from the neural network
            policy_state_value, policy_advantage_value = policy_network.forward(state_tensor)
            target_state_value, target_advantage_value = target_network.forward(new_state_tensor)

            #calculate the q_values of the performed actions according to the (Wang et al, 2016) paper - https://arxiv.org/pdf/1511.06581.pdf
            policy_advantage_deviation        = (policy_advantage_value - policy_advantage_value.mean(dim=1, keepdim=True))
            sum_of_policy_value_and_advantage = torch.add(policy_state_value, policy_advantage_deviation)
            q_values_of_policy_actions        = sum_of_policy_value_and_advantage.gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)

            #calculate the q_values of the best future actions.
            target_advantage_deviation = (target_advantage_value - target_advantage_value.mean(dim=1, keepdim=True))
            q_values_of_target_actions = torch.add(target_state_value, target_advantage_deviation)

            #using the standard bellman equation
            q_values_to_update_to = reward_tensor + gamma * torch.max(q_values_of_target_actions, dim=1)[0].detach()

            print(q_values_of_policy_actions)
            print(q_values_to_update_to)
            #Calculate the loss to prioritize the experience replay and store the priority in the experience replay
            loss = (q_values_of_policy_actions.detach() - q_values_to_update_to).pow(2) * weights
            priorities_of_q_values = loss + 1e-5
            print(priorities_of_q_values)
            self.updatePriorities(sample_indices, priorities_of_q_values)

            #calculate the loss and backpropagate
            lossFunction = F.mse_loss(q_values_to_update_to, q_values_of_policy_actions)
            optimizer.zero_grad()
            lossFunction.backward()
            optimizer.step()

            return
        if double_dueling_q_learning:

            #retrieve values of the best future action according to the policy network
            policy_future_state_value, policy_future_advantage_value    = policy_network.forward(new_state_tensor)
            policy_future_advantage_deviation                           = (policy_future_advantage_value - policy_future_advantage_value.mean(dim=1, keepdim=True))
            q_values_of_future_policy_actions                           = torch.add(policy_future_state_value, policy_future_advantage_deviation)
            best_action_indices_policy_network_new_state                = q_values_of_future_policy_actions.argmax(dim=1)

            #retrieve values of the best future actions from target network according to the the policy network
            target_state_value, target_advantage_value       = target_network.forward(new_state_tensor)
            target_future_advantage_deviation                = (target_advantage_value - target_advantage_value.mean(dim=1, keepdim=True))
            q_values_of_future_target_actions                = torch.add(target_state_value, target_future_advantage_deviation)
            target_network_selected_future_advantages_values = q_values_of_future_target_actions[batch_index, best_action_indices_policy_network_new_state]
            q_values_to_update_to                            = reward_tensor + gamma * target_network_selected_future_advantages_values
            q_values_to_update_to[terminal_tensor]           = 0.0 #set terminal states of the environment to zero as we do not want to learn from this

            #retrieve the values of the best current state action pairs according to the policy network
            policy_state_value, policy_advantage_value = policy_network.forward(state_tensor)
            policy_advantage_deviation                 = (policy_advantage_value - policy_advantage_value.mean(dim=1, keepdim=True))
            q_values_of_policy_actions                 = torch.add(policy_state_value, policy_advantage_deviation)
            q_values_selected_actions_policy_values    = q_values_of_policy_actions[batch_index, action_tensor]

            lossFunction = F.mse_loss(q_values_to_update_to, q_values_selected_actions_policy_values)
            optimizer.zero_grad()
            lossFunction.backward()
            optimizer.step()

            return
        if double_q_learning:
            #Retrieve the predicted q-values of the next states and select the best action according to policy_network
            policy_network_q_values         = policy_network.forward(new_state_tensor)
            policy_network_q_values_indeces = policy_network_q_values.argmax(dim=1)

            #Predict the q_values according to the target_network and select the q_values from the best actions according to policy_network
            target_network_q_values                = target_network.forward(new_state_tensor)
            target_network_selected_q_values       = target_network_q_values[batch_index, policy_network_q_values_indeces]
            q_values_to_update_to                  = reward_tensor + gamma * target_network_selected_q_values
            q_values_to_update_to[terminal_tensor] = 0.0 #If a Q-values is from a terminal state, make sure not to use this Q-value

            #Predict the q_values of the current state according to the policy network
            policy_network_q_values_current_state                  = policy_network.forward(state_tensor)
            policy_network_q_values_selected_actions_current_state = policy_network_q_values_current_state[batch_index, action_array]

            lossFunction = F.mse_loss(q_values_to_update_to, policy_network_q_values_selected_actions_current_state)
            optimizer.zero_grad()
            lossFunction.backward()
            optimizer.step()

            return

        if deep_q_learning:
            q_values_current_state  = policy_network.forward(state_tensor)[batch_index, action_array]
            q_values_next_state     = target_network.forward(new_state_tensor)
            q_values_next_state[terminal_tensor] = 0.0 #if the next q-value represents a terminal state we do not want to learn from this, hence we set the q-value to 0

            q_values_next_state = (reward_tensor * gamma) + torch.max(q_values_next_state, dim=1)[0]

            #Determine the loss and do backpropagation on the policy network
            lossFunction = F.mse_loss(q_values_current_state, q_values_next_state)
            optimizer.zero_grad()
            lossFunction.backward()
            optimizer.step()

            return
def plot(x_axis, y_axis):
    plt.figure(2)
    plt.clf()
    #plt.title('Training...')
    plt.xlabel('Amount of times target network updated from policy network x10.000')
    plt.ylabel('Reward per episode')
    plt.plot(x_axis, y_axis)
    #plt.plot(reward_avg, label="Times policy network updated from target network x100.000")
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", borderaxespad=0.)
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
#Determine the type of Q-learning to be applied to the network
double_q_learning         = False
deep_q_learning           = False
dueling_q_learning        = True
double_dueling_q_learning = False

#Determine constants for the agent
BATCH_SIZE          = 50           #Batches we want to update our policy network from
GAMMA               = 0.99         #The discount factor for every step that is taken.
EPSILON_START       = 1            #The epsilon value we want to start with indicating exploration
EPSILON_END         = 0.01         #The value for epsilon when we want to stop explore
EPSILON_DECAY       = 0.001        #How fast will epsilon_decay?
TARGET_UPDATE       = 6            #The amount of episodes after which we want to
MEMORY_SIZE         = 100000       #Capacity for the memory and thus the amount we want to store
AMOUNT_OF_EPISODES  = 1000        #The amount of episodes we want to play the game for training the network
LEARNING_RATE       = 0.001

#Setup the device, the environmentManager and the agent
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environmentManager      = lunarEnvironmentManager(device)
agent                   = lunarLanderAgent(environmentManager.amount_of_actions()\
                                           , device, EPSILON_START, EPSILON_END, EPSILON_DECAY)

#make a target and policy network. We copy all weights and biases of the policy network to the target network.
policy_network          = neuralNetwork(environmentManager.amount_of_state_inputs(), environmentManager.amount_of_actions(), dueling_q_learning, double_dueling_q_learning).to(device)
target_network          = neuralNetwork(environmentManager.amount_of_state_inputs(), environmentManager.amount_of_actions(), dueling_q_learning, double_dueling_q_learning).to(device)
target_network.load_state_dict(policy_network.state_dict()) #Copy weights and biases of policy network to target network
target_network.eval()
episodes_for_target_network_update  = 5

#Initialize the replayMemory and the optimizer. In this case we use Adam as an optimizer. There is no real reason to use Adam, just because it's the standard
replayMemory            = replayMemory(MEMORY_SIZE,device,environmentManager.amount_of_state_inputs(), BATCH_SIZE)
optimizer               = optim.Adam(params=policy_network.parameters())

#Initialize some variables for plotting the progress of the agent
total_reward_for_episodes          = []
reward_average                     = [0]
amount_of_optimization_steps       = 0
amount_of_optimization_steps_array = []

for episode in range(AMOUNT_OF_EPISODES):
    total_reward_for_episode    = 0
    state                       = torch.tensor(environmentManager.reset())
    print(episode)
    while not environmentManager.done:
        action                        = agent.select_action(state, policy_network, dueling_q_learning, double_dueling_q_learning) #select an action according to highest q-value retrieved from the policy network
        new_state, reward, done       = environmentManager.take_action_and_observe(action)
        new_state = torch.tensor(new_state).to(device)
        replayMemory.pushExperienceInMemory(state, new_state, reward, action, done)
        state = new_state
        total_reward_for_episode += reward

        #if episode > (AMOUNT_OF_EPISODES-10):
        #environmentManager.render()

        if replayMemory.canMemoryProvideSampleOfBatchSize(BATCH_SIZE):
            replayMemory.learnFromMemory(device, policy_network, target_network, BATCH_SIZE, optimizer, GAMMA, double_q_learning, deep_q_learning, dueling_q_learning, double_dueling_q_learning)
            amount_of_optimization_steps += 1

    print("Reward for episode")
    print(total_reward_for_episode)
    target_network_update_needed = episode % episodes_for_target_network_update
    if target_network_update_needed == 0:
        print("updating target network")
        target_network.load_state_dict(policy_network.state_dict())
        target_network.eval()
    #amount_of_optimization_steps /= 10000
    amount_of_optimization_steps_array.append(amount_of_optimization_steps)
    total_reward_for_episodes.append(total_reward_for_episode)
    reward_average.append(sum(total_reward_for_episodes)/(len(total_reward_for_episodes)))

plot(amount_of_optimization_steps_array, total_reward_for_episodes)
environmentManager.close()