import sys

import dqn
import numpy as np
import torch
class dqnAgent:
    # non c'è epsilon perchè l'azione da compiere viene scelta deterministicamente, non serve sceglierla in modo epsilon-greedy
    # gamma rappresenta il discount factor dei reward futuri. In questo caso sarà pari a e^(-r * delta t), con delta t la differenza tra le due date in cui è possibile esercitare l'opzione
    # max_memory rappresenta la dimensione della rete di memoria
    def __init__(self, gamma, batch_size, epsilon, n_actions, max_mem_size=100000, epsilon_end = 0.01, epsilon_decay = 0.001):
        self.gamma = gamma
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.action_space = [i for i in range(n_actions)]
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.mem_counter = 0
        self.Q_eval = dqn.DeepQNetwork(input_dims=[4], hidden_layer_dims=4, output_layer_dims=4, n_actions=2)
        self.state_memory = np.zeros((self.mem_size, 4), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 4), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        #Store a transition in the replay memory
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def choose_action(self, observation):
        # epsilon greedy strategy
        a = np.random.random()
        # print("epsilon: ",self.epsilon)
        # print("a: ",a)
        if a > self.epsilon:
            # print("sono nell'if ")
            state = torch.tensor(observation, dtype=torch.float32)
            state = state.expand(4, 4)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            # print("obs 1: ",observation[1], "obs 0: ", observation[0])
            # print((observation[1] > observation[0]))
            if(observation[1] > observation[0]):
                action = 1
            else:
                action = 0
        return action

    def learn(self, action):
        if self.mem_counter < self.batch_size:
            return -1
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_memory[batch], dtype=torch.float32) #da numpy array a pytorch tensor
        new_state_batch = torch.tensor(self.new_state_memory[batch], dtype=torch.float32) #da numpy array a pytorch tensor
        reward_batch = torch.tensor(self.reward_memory[batch], dtype=torch.float32) #da numpy array a pytorch tensor
        terminal_batch = torch.tensor(self.terminal_memory[batch], dtype=torch.bool) #da numpy array a pytorch tensor
        action_batch = self.action_memory[batch] #questo può rimanere un numpy array
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]  #valore della q function per lo stato corrente
        q_next = self.Q_eval.forward(new_state_batch) #valore della q function stimato per lo stato successivo
        print("nuovi stati: ", new_state_batch)
        print("q_eval: ", q_eval)
        print("q_next: ", q_next)
        # print("reward_batch: ",reward_batch)
        q_next[terminal_batch] = 0.0
        if(action == 1):
            q_target = reward_batch
        else:
            q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] #[0] è il valore massimo, perchè torch.max ritorna sia il valore massimo che l'indice del massimo
        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_end
        return loss.item()

    def set_parameters(self, netParams):
        hidden_layer_weights = np.float32(netParams.previous_layer.trained_weights)
        hidden_layer_weights = hidden_layer_weights.copy()
        output_layer_weights = np.float32(netParams.trained_weights)
        output_layer_weights = output_layer_weights.copy()

        for name, param in self.Q_eval.named_parameters():
            # setto i pesi solo per l'hidden layer e l'output layer
            if name.startswith("2."):
                param.data = torch.from_numpy(hidden_layer_weights)  # converto da ndarray a pytorch tensor
            if name.startswith("4."):
                param.data = torch.from_numpy(output_layer_weights)  # converto da ndarray a pytorch tensor