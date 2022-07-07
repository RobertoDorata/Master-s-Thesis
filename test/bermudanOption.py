import sys

import bermudanOptionEnvironment
import dqn
import dqnAgent
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

class bermudanOption:
    def __init__(self):
        self.env = bermudanOptionEnvironment.bermudanOptionEnv(1, 0.05, 38, 0, 38*365)
        self.agent = dqnAgent.dqnAgent(gamma=np.exp(-0.05 * 1 / 252), batch_size=1, epsilon=1.0, n_actions=2)

    def getScore(self, netParams):
        score = 0
        giorniPassati = 0
        loss = 1000000
        array_loss = []
        # exTimes = [153, 195, 94, 338, 76, 310, 42, 224, 200, 213]  # giorni dell'anno in cui sarà possibile esercitare la bermudan option
        # K = 110.0
        # Create the PyTorch model.
        # hidden_layer_weights = np.float32(netParams.previous_layer.trained_weights)
        # hidden_layer_weights = hidden_layer_weights.copy()
        # output_layer_weights = np.float32(netParams.trained_weights)
        # output_layer_weights = output_layer_weights.copy()
        #
        # model = dqn.DeepQNetwork([4], 4, 4, 2)
        #
        # for name, param in model.named_parameters():
        #     #setto i pesi solo per l'hidden layer e l'output layer
        #     if name.startswith("2."):
        #         param.data = torch.from_numpy(hidden_layer_weights) #converto da ndarray a pytorch tensor
        #     if name.startswith("4."):
        #         param.data = torch.from_numpy(output_layer_weights) #converto da ndarray a pytorch tensor

        self.agent.set_parameters(netParams)
        observation = self.env.reset()
        while True:
            action = self.agent.choose_action(observation)
            new_observation, reward, done, info = self.env.step(action, giorniPassati)
            score += reward
            if done:
                break
            else:
                self.agent.store_transition(observation, action, reward, new_observation, done)
                loss = self.agent.learn(action)
                array_loss.append(loss)
                giorniPassati += 1
            observation = new_observation
        return 1/loss

    def saveParams(self, netParams):
        """
        serializes and saves a list of network parameters using pickle
        :param netParams: a list of floats representing the network parameters (weights and biases) of the MLP
        """

        pickle.dump(netParams, open("cart-pole-data.pickle", "wb"))

    def replayWithSavedParams(self, netParams):
        score = 0
        giorniPassati = 0
        self.agent.set_parameters(netParams)
        observation = self.env.reset()
        while True:
            action = self.agent.choose_action(observation)
            new_observation, reward, done, info = self.env.step(action, giorniPassati)
            score += reward
            giorniPassati += 1
            observation = new_observation
            if done:
                break
        print("score: ",score)

    def replay(self, netParams):
        """
        renders the environment and uses the given network parameters to replay an episode, to visualize a given solution
        :param netParams: a list of floats representing the network parameters (weights and biases) of the MLP
        """
        mlp = self.initMlp(netParams)

        self.env.render()

        actionCounter = 0
        totalReward = 0
        observation = self.env.reset()
        action = int(mlp.predict(observation.reshape(1, -1)) > 0)

        while True:
            actionCounter += 1
            self.env.render()
            observation, reward, done, info = self.env.step(action)
            totalReward += reward

            print(actionCounter, ": --------------------------")
            print("action = ", action)
            print("observation = ", observation)
            print("reward = ", reward)
            print("totalReward = ", totalReward)
            print("done = ", done)
            print()

            if done:
                break
            else:
                time.sleep(0.03)
                action = int(mlp.predict(observation.reshape(1, -1)) > 0)
