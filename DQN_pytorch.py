import pickle
import numpy as np
import gym
import torch
from tqdm import tqdm
import OpenAiGymEnvironment
import DQNAgent
import matplotlib.pyplot as plt
import pygad.torchga as torchga
import pygad

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)
    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error

    return solution_fitness

def callback_generation(ga_instance):
    #print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print()

input_layer = torch.nn.Linear(3, 2)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(2, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.L1Loss()

# Data inputs
data_inputs = torch.tensor([[0.02, 0.1, 0.15],
                            [0.7, 0.6, 0.8],
                            [1.5, 1.2, 1.7],
                            [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = torch.tensor([[0.1],
                             [0.6],
                             [1.3],
                             [2.5]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 250 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
#print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                      weights_vector=solution)
model.load_state_dict(best_solution_weights)
predictions = model(data_inputs)
#print("Predictions : \n", predictions.detach().numpy())

abs_error = loss_function(predictions, data_outputs)
#print("Absolute Error : ", abs_error.detach().numpy())



env = OpenAiGymEnvironment.AmeriOptionEnv()
observation_space = env.observation_space.shape
action_space = 2

training_mode = True
num_episodes = 150
loss = []
pretrained = False

exTimes = [153, 195, 94, 338, 76, 310, 42, 224, 200, 213]
S0 = 100.0
K = 110.0
r = 0.1
T = 1.0
# Create the PyTorch model.
input_layer = torch.nn.Linear(2, 128)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(128, 128)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)

train_step_counter = torch.tensor(0)

agent = DQNAgent.DQNAgent(state_space=observation_space,
                            action_space=action_space,
                            model=model,
                            max_memory_size=30000,
                            batch_size=128,
                            gamma=0.8,
                            lr=0.0001,
                            dropout=0.1,
                            exploration_max=1.0,
                            exploration_min=0.02,
                            exploration_decay=0.99,
                            pretrained=pretrained,
                            K=K,
                            r=r,
                            T=T,
                            exTimes=exTimes)

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
    """
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.plot(sim_prices)
    plt.show()
    """
    state = torch.Tensor([state])
    total_reward = 0
    steps = 0
    while True:
        action = agent.act(state, steps, sim_prices)
        print("action: ",action)
        print("step: ", steps)
        steps += 1

        state_next, reward, terminal, info = env.step(action)
        total_reward += reward
        state_next_numpy_array = np.array([state_next])
        state_next = torch.Tensor(state_next_numpy_array)
        reward = torch.tensor([reward]).unsqueeze(0)

        terminal = torch.tensor([int(terminal)]).unsqueeze(0)

        if training_mode:
            agent.remember(state, action, reward, state_next, terminal)
            loss.append(agent.experience_replay())
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

    loss = [i for i in loss if i < 20]
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()
env.close()
