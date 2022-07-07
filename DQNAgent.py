import pickle
import torch
import torch.nn as nn
import random
import DeepQNetwork
import numpy as np


class DQNAgent:
    # non c'è epsilon perchè l'azione da compiere viene scelta deterministicamente, non serve sceglierla in modo epsilon-greedy
    # gamma rappresenta il discount factor dei reward futuri. In questo caso sarà pari a e^(-r * delta t), con delta t la differenza tra le due date in cui è possibile esercitare l'opzione
    # max_memory rappresenta la dimensione della rete di memoria
    def __init__(self, gammma, input_dims, batch_size, n_actions, max_mem_size=100000):
        self.gamma = gammma
        self.K = 100
        self.r = 0.016
        self.T = 1.0
        self.exTimes = [10,100]
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0 #tiene conto della prima posizione disponibile in cui poter salvare la memoria dell'agente
        input_layer = torch.nn.Linear(2, 128)
        relu_layer = torch.nn.ReLU()
        output_layer = torch.nn.Linear(128, 128)

        self.Q_eval = DeepQNetwork.DeepQNetwork(input_dims, hidden_layer_dims=4, output_layer_dims=4, n_actions=n_actions)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state2, done):
        #Store a transition in the replay memory
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state2
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        #se esercito
        state = torch.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = torch.argmax(actions).item()
        #se non esercito
        action = 0

        return action

    def learn(self):
        #usando l'algoritmo genetico, imposta i pesi della neural network
        if self.mem_cntr < self.batch_size:
            return

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]


        # DQN network
        self.dqn = model

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            with open("ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open("num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state, steps, sim_prices):
        action = 0
        print('steps: ', steps)
        print('state: ', state)
        # posso esercitare solo nelle exTimes, o alla maturity
        if (steps in self.exTimes or steps == 365):
            action = 1
        # non sono nelle exTimes, vedo se sono almento alla maturity
        return action

    def experience_replay(self):
        # print('memory sample size: ', self.memory_sample_size)
        # print('num in queue: ', self.num_in_queue)
        if self.memory_sample_size > self.num_in_queue:
            return 100

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        # se l'azione è stop
        if (ACTION == 1):
            target = REWARD
        else:
            # se l'azione è esercitare
            target = REWARD + self.gamma * torch.max(self.dqn(STATE2, 1), self.dqn(STATE2, 0))
            # q function
        # target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        # print('loss: ', loss.item())
        loss.backward()  # Compute gradients

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
        return loss.item()
