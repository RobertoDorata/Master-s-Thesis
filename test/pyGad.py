import sys
import pygad
import pygad.gann as gann
import bermudanOption


#bermudanOption = bermudanOption.bermudanOption()
#bermudanOption.replayWithSavedParams()
#sys.exit()



#la fitness function sarà la loss ottenuta in un episodio
def fitness_func(solution, sol_idx):
    global GANN_instance
    layers_weights = GANN_instance.population_networks[sol_idx]
    solution_fitness = bermudanOption.getScore(layers_weights)
    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)
    last_fitness = ga_instance.best_solution()[1].copy()
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))

#create an instance of the neural network, of the type multilayer perceptron, and make the training using the PyGad library
num_solutions = 2#10 #Number of neural networks (i.e. solutions) in the population.
num_neurons_input = 4
num_neurons_hidden_layers = [4] #lista di 1 solo elemento perchè c'è un solo hidden layer
num_neurons_output_layer = 4
hidden_activations = ["relu"]
GANN_instance = gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_neurons_input,
                                num_neurons_hidden_layers=num_neurons_hidden_layers,
                                num_neurons_output=num_neurons_output_layer,
                                hidden_activations=hidden_activations)

population_vectors = gann.population_as_vectors(population_networks=GANN_instance.population_networks)
bermudanOption = bermudanOption.bermudanOption()
last_fitness = 0

initial_population = population_vectors.copy()

num_parents_mating = 2#4

num_generations = 1#5

mutation_percent_genes = 5

parent_selection_type = "tournament"

crossover_type = "single_point"

mutation_type = "random"

keep_parents = 1

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       K_tournament=3,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()
ga_instance.plot_fitness()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
layers_weights = GANN_instance.population_networks[solution_idx]
print("migliore: ", layers_weights)

print("Parameters of the best solution : {solution}".format(solution=solution)) #questi saranno poi i parametri della rete neurale che verranno usati nel dqn per approssimare la q-function
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

bermudanOption.saveParams(layers_weights)

