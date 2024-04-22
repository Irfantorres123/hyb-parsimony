import random
import numpy as np
from pyDOE2 import lhs

num_hyperparameters = 5
num_features = 5
min_param = np.array([0]*10)  # Adjust this based on the actual range of your parameters
max_param = np.array([1]*10)  # Adjust this as well
elite_size = 10
population_size = 50
generations = 30
tournament_size = 5
mutation_rate = 0.01
    
def create_individual(num_hyperparameters, num_features, min_param, max_param):
    hyperparameters = lhs(num_hyperparameters, samples=1) * (max_param[: num_hyperparameters] - min_param[: num_hyperparameters]) + min_param[: num_hyperparameters]
    features = lhs(num_features, samples=1) * (max_param[num_hyperparameters:] - min_param[num_hyperparameters:]) + min_param[num_hyperparameters:]
    # Ensure hyperparameters and features are flattened before concatenation if they aren't already
    hyperparameters = hyperparameters.flatten()
    features = features.flatten()
    
    # Concatenate the two parts and return
    return np.concatenate([hyperparameters, features])

def internal_complexity_measure(individual):
    # Need to think about how it can be done for other model as well
    # Define complexity based on model specifics
    return np.sum(np.square(individual))# Example: sum of squares for regression coefficients

def fitness_and_complexity(individual):
    #individual[:num_hyperparams] are hyperparameters and the rest are features
    J = np.sum(individual)
    Mc = 1000000 * np.count_nonzero(individual[num_hyperparameters:] < 0.5) + internal_complexity_measure(individual)
    return J, Mc

def sort_and_promote(population):
    # Assuming population is a numpy array and fitness_and_complexity returns a tuple where the first element is what you want to sort by
    indices = np.argsort([fitness_and_complexity(ind)[0] for ind in population])
    return population[indices]  # This sorts the array according to the indices from argsort

def tournament_selection(soln_space, tournament_size):
    # Convert numpy array to list if it's not already a list
    if isinstance(soln_space, np.ndarray):
        soln_space = soln_space.tolist()
    sample = random.sample(soln_space, tournament_size)
    sample.sort(key=lambda x: fitness_and_complexity(x)[0])  # Sort based on fitness 'J'
    return sample[0]

# Crossover between two parents to create two children
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def create_generation(population_size, current_generation, tournament_size, mutation_rate):
    next_generation = []
    while len(next_generation) < population_size:
        parent1 = tournament_selection(current_generation, tournament_size)
        parent2 = tournament_selection(current_generation, tournament_size)
        child1, child2 = crossover(parent1, parent2)
        next_generation.append(mutate(child1, mutation_rate))
        if len(next_generation) < population_size:
            next_generation.append(mutate(child2, mutation_rate))
    return next_generation

def elite_production(current_generation, elite_size, population_size, tournament_size, mutation_rate):
    sorted_population = sort_and_promote(current_generation)
    elite_population = sorted_population[:elite_size]
    non_elite_population = create_generation(population_size - elite_size, sorted_population[elite_size:], tournament_size, mutation_rate)
    if isinstance(elite_population, list):
        return elite_population + non_elite_population.tolist()  # Convert to list if needed
    return np.concatenate([elite_population, non_elite_population])

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = individual[i] + np.random.normal(0, 0.1)
    return individual

def train_validate(individual):
    model = train_model(individual)
    validation_score = validate_model(model)
    return validation_score

def genetic_algorithm(generations, population_size, num_hyperparameters, num_features, min_param, max_param, elite_size, tournament_size, mutation_rate):
    soln_space = np.array([create_individual(num_hyperparameters, num_features, min_param, max_param) for _ in range(population_size)])
    for generation in range(generations):
        soln_space = sort_and_promote(soln_space)
        soln_space = elite_production(soln_space, elite_size, population_size, tournament_size, mutation_rate)
        
        best_fitness = min(fitness_and_complexity(ind)[0] for ind in soln_space)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        
    best_individual = min(soln_space, key=lambda x: fitness_and_complexity(x)[0])
    return best_individual

if __name__ == "__main__":
    best_solution = genetic_algorithm(generations, population_size, num_hyperparameters, num_features, min_param, max_param, elite_size, tournament_size, mutation_rate)
    print("Best Solution:", best_solution)
    print("Fitness of Best Solution:", fitness_and_complexity(best_solution)[0])  # Display the fitness part of the tuple

    