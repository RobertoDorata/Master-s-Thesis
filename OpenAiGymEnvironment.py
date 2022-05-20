import numpy as np
import gym
# per una put bermudana con maturity T=1, K=110, S0=100, sigma=0.25, esercitabile in N=10 date differenti, con r=0.1, si ha un prezzo pari a 11.987
class AmeriOptionEnv(gym.Env):
    def __init__(self):
        self.S0 = 100.0
        self.K = 110.0
        self.r = 0.1
        self.sigma = 0.25
        self.T = 1.0
        self.N = 365  # 365 day

        self.S1 = 0
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N

        self.action_space = gym.spaces.Discrete(2)  # 0: hold, 1:exercise
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1.0]),
                                                dtype=np.float32)  # S in [0, inf], tao in [0, 1]

    def step(self, action):
        if action == 1:  # exercise
            #print('esercitata: ', str(self.S1))
            reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T * (self.day_step / self.N))
            #print('reward: ', reward)
            done = True
        else:  # hold
            if self.day_step == self.N:  # at maturity
                reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T)
                #print('sono alla maturity, e ho un reward di: ', reward)
                done = True
            else:  # move to tomorrow
                reward = 0
                # lnS1 - lnS0 = (r - 0.5*sigma^2)*t + sigma * Wt
                self.S1 = self.S1 * np.exp((self.r - 0.5 * self.sigma ** 2) * (self.T / self.N) + self.sigma * np.sqrt(
                    self.T / self.N) * np.random.normal())
                #print('non ho esercitato: S1 vale ', self.S1)
                self.day_step += 1
                done = False

        tao = 1.0 - self.day_step / self.N  # time to maturity, in unit of years
        return np.array([self.S1, tao]), reward, done, {}

    def reset(self):
        self.day_step = 0
        self.S1 = self.S0
        tao = 1.0 - self.day_step / self.N  # time to maturity, in unit of years
        return [self.S1, tao]

    def render(self):
        """
        make video
        """
        pass

    def close(self):
        pass