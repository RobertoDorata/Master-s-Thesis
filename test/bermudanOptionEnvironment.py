import numpy as np
from matplotlib import pyplot as plt

class bermudanOptionEnv:
    def __init__(self, K, r, T, T0, T1):
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N
        self.K = K
        self.r = r
        self.T = T
        self.T0 = T0
        self.T1 = T1
        self.done = False
        self.action_space = [0, 1] # 0: hold, 1:exercise
        self.S1 = []
        self.info = []
        self.observation = np.array([0, 0, 0.0, 0.0])
        # plt.xlabel('Date')
        # plt.ylabel('Stock Price')
        # plt.plot(self.S1)
        # plt.show()

    def step(self, action, i):
        # stop
        if action == 1 or i == ((38*365)-1):
            # il discount factor per un'opzione bermmudana è dato da e^(-r * delta t), con delta t la differenza tra le due date in cui è possibile esercitare
            self.reward = max(self.K - self.S1[i], 0.0) # payoff di una put option
            self.done = True
            self.observation = np.array([self.S1[i], 1, 38, 38*365])
            # observation: il valore del sottostante e il valore dell'opzione
            # vedere se aggiungere anche il continuation value dell'opzione
            # va aggiunta anche la prossima data in cui è possibile esercitare l'opzione, e la data precedente
            # va aggiunta anche la maturity delineation
        # continue
        else:
            self.reward = 0

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N
        self.done = False
        self.action_space = [0, 1]  # 0: hold, 1:exercise
        self.S1 = self.simulateUnderlyingPrice(1, 0.05, 38, 0.2, 1, 38*365)[1]
        return np.array([self.S1[0], 1, 0.419, 153.0])

    def simulateUnderlyingPrice(self, S0, r, T, sigma, M, I):
        dt = float(T) / M
        paths = np.zeros((M + 1, I), np.float64)
        paths[0] = S0
        for t in range(1, M + 1):
            rand = np.random.standard_normal(I)
            paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                             sigma * np.sqrt(dt) * rand)
            paths[t] = paths[t]
        return paths