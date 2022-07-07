import numpy as np
# per una put bermudana con maturity T=1, K=110, S0=100, sigma=0.25, esercitabile in N=10 date differenti, con r=0.1, si ha un prezzo pari a 11.987
class AmeriOptionEnv():
    def __init__(self):
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N
        self.done = False
        self.action_space = [0, 1] # 0: hold, 1:exercise
        self.observation_space = []

    def step(self, action):
        if action == 0: #continue
            self.reward = 0
        else: #stop
            #il discount factor per un'opzione bermmudana è dato da e^(-r * delta t), con delta t la differenza tra le due date in cui è possibile esercitare
            self.reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T * (self.day_step / self.N)) #payoff di una put option
            self.done = True
            #observation: il valore del sottostante e il valore dell'opzione
            #vedere se aggiungere anche il continuation value dell'opzione
            #va aggiunta anche la prossima data in cui è possibile esercitare l'opzione, e la data precedente
            #va aggiunta anche la maturity dell'opzione
        # if action == 1:  # exercise
        #     #print('esercitata: ')
        #     self.reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T * (self.day_step / self.N))
        #     #print('reward: ', reward)
        #     self.done = True
        # elif action == 0:  # hold
        #     if self.day_step == self.N:  # at maturity
        #         self.reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T)
        #         #print("sono alla maturity")
        #         self.done = True
        #     else:  # move to tomorrow
        #         self.reward = 0
        #         # lnS1 - lnS0 = (r - 0.5*sigma^2)*t + sigma * Wt
        #         self.S1 = self.S1 * np.exp((self.r - 0.5 * self.sigma ** 2) * (self.T / self.N) + self.sigma * np.sqrt(
        #             self.T / self.N) * np.random.normal())
        #         #print('non ho esercitato: S1 vale ', self.S1)
        #         self.day_step += 1
        #         self.done = False
        #
        return np.array([self.S1, tao]), self.reward, self.done, {}

    def reset(self):
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N
        self.done = False
        self.action_space = [0, 1]  # 0: hold, 1:exercise
        self.observation_space = []
        return np.concatenate(([self.position_value, self.remaining_time], self.observations[0]))

    def simulateUnderlyingPrice(self, S0, r, T, sigma, M, I):
        dt = float(T) / M
        paths = np.zeros((M + 1, I), np.float64)
        paths[0] = S0
        for t in range(1, M + 1):
            rand = np.random.standard_normal(I)
            paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                             sigma * np.sqrt(dt) * rand)
        return paths

