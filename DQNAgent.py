import pickle
import torch
import torch.nn as nn
import random
import numpy as np

class DQNAgent:

    def __init__(self, state_space, action_space, model, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, pretrained,K,r,T,exTimes):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.K = K
        self.r = r
        self.T = T
        self.exTimes = exTimes

        # DQN network
        self.dqn = model

        if self.pretrained:
            self.dqn.load_state_dict(torch.load("DQN.pt", map_location=torch.device(self.device)))
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

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
        """Epsilon-greedy action
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        """
        actual_value = max(self.K - sim_prices[steps], 0.0) * np.exp(-self.r * self.T * (steps / 365))
        continuation_value = 0
        action = 0
        #posso esercitare solo nelle exTimes, o alla maturity
        if(steps in self.exTimes):
            print("posso esercitare, sono alla data ", steps)
            for i in range(self.exTimes.index(steps), len(sim_prices)):
                next_price = sim_prices[i]
                continuation_value += max(self.K - next_price, 0.0) * np.exp(-self.r * self.T * (i / 365))
            if(actual_value > continuation_value):
                #esercito
                action = 1
        #non sono nelle exTimes, vedo se sono almento alla maturity
        else:
            if(steps == 365):
                continuation_value = max(self.K - sim_prices[-1], 0.0) * np.exp(-self.r * self.T)
                if(actual_value > continuation_value):
                    # esercito
                    action = 1
            #non sono ne alle ex times ne alla maturity, non posso esercitare
            else:
                print("non posso esercitare, sono alla data ",steps)
        print("actual: ", actual_value)
        print("continuation: ", continuation_value)
        print("action: ", action)
        return action

    def experience_replay(self):
        #print('memory sample size: ', self.memory_sample_size)
        #print('num in queue: ', self.num_in_queue)
        if self.memory_sample_size > self.num_in_queue:
            return 100

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        #print('loss: ', loss.item())
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
        return loss.item()