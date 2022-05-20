import pickle
import numpy as np
import gym
import torch
from tqdm import tqdm
import OpenAiGymEnvironment
import DQNAgent
import matplotlib.pyplot as plt

env = OpenAiGymEnvironment.AmeriOptionEnv()
observation_space = env.observation_space.shape
action_space = 2

training_mode = True
num_episodes=2000
pretrained = False

exTimes = [153, 195, 94, 338, 76, 310, 42, 224, 200, 213]

# Create the PyTorch model.
input_layer = torch.nn.Linear(2, 12)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(12, 2)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)

train_step_counter = torch.tensor(0)

agent = DQNAgent.DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     model=model,
                     max_memory_size=30000,
                     batch_size=12,
                     gamma=0.9,
                     lr=0.001,
                     dropout=0.2,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     pretrained=pretrained)

total_rewards = []
if training_mode and pretrained:
    with open("total_rewards.pkl", 'rb') as f:
        total_rewards = pickle.load(f)

for ep_num in tqdm(range(num_episodes)):
    state = env.reset()
    sim_prices = [state[0]]
    for i in range(365):
        action = 0
        s_next, reward, done, info = env.step(action)
        sim_prices.append(s_next[0])
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.plot(sim_prices)
    #plt.show()

    state = torch.Tensor([state])
    total_reward = 0
    steps = 0
    while True:
        action = agent.act(state)
        steps += 1

        state_next, reward, terminal, info = env.step(int(action[0]))
        total_reward += reward
        state_next_numpy_array = np.array([state_next])
        state_next = torch.Tensor(state_next_numpy_array)
        reward = torch.tensor([reward]).unsqueeze(0)

        terminal = torch.tensor([int(terminal)]).unsqueeze(0)

        if training_mode:
            agent.remember(state, action, reward, state_next, terminal)
            agent.experience_replay()

        state = state_next
        if terminal:
            break

    total_rewards.append(total_reward)

    if ep_num != 0: #and ep_num % 100 == 0:
        print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))
    num_episodes += 1

# Save the trained memory so that we can continue from where we stop using 'pretrained' = True
if training_mode:
    with open("ending_position.pkl", "wb") as f:
        pickle.dump(agent.ending_position, f)
    with open("num_in_queue.pkl", "wb") as f:
        pickle.dump(agent.num_in_queue, f)
    with open("total_rewards.pkl", "wb") as f:
        pickle.dump(total_rewards, f)

    torch.save(agent.dqn.state_dict(), "DQN.pt")
    torch.save(agent.STATE_MEM, "STATE_MEM.pt")
    torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
    torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
    torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
    torch.save(agent.DONE_MEM, "DONE_MEM.pt")

env.close()