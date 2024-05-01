import numpy as np
import random

seed = 1990 #123 #1234 #69
np.random.seed(seed)

# Configurations
num_trials = 30
generations = 30
population_size = 50
length = 2 #The number of elements (genes) in each individual -- dimensions
tournament_size = 5
mutation_rate = 0.05

def rosenbrock(x):
    # Calculate the value of the rosenbrock's function
    x1 = x[0]
    x2 = x[1]
    return (x1 - 1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def easom_function(x):
    # Compute Easom function value
    x1 = x[0]
    x2 = x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x1 - np.pi)**2)

def fitness(individual):
    return sum(individual)

def get_individual(length):
    return [random.random() for _ in range(length)]

def create_population(population_size, length):
    return [get_individual(length) for _ in range(population_size)]

def tournament_selection(f, soln_space, tournament_size):
    sample = random.sample(soln_space, tournament_size)
    sample.sort(key=lambda x: f(x))
    return sample[0]

# Crossover between two parents to create two children
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) -1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation function to introduce random changes in an individual
def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.random()
    return individual

def create_generation(f, population_size, current_generation, tournament_size, mutation_rate):
    next_generation = []
    while len(next_generation) < population_size:
        parent1 = tournament_selection(f, current_generation, tournament_size)
        parent2 = tournament_selection(f, current_generation, tournament_size)
        child1, child2 = crossover(parent1, parent2)
        next_generation.append(mutate(child1, mutation_rate))
        if len(next_generation) < population_size:
            next_generation.append(mutate(child2, mutation_rate))
    
    return next_generation
    
def genetic_algorithm(f, generations, population_size, length, tournament_size, mutation_rate):
    soln_space = create_population(population_size, length)
    for generation in range(generations):
        soln_space = create_generation(f, population_size, soln_space, tournament_size, mutation_rate)
        best_fitness = min(f(ind) for ind in soln_space)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
     
    best_individual = min(soln_space, key=lambda x:f(x))
    return best_individual

# Example usage of the genetic algorithm
if __name__ == "__main__":
    best_solution = genetic_algorithm(rosenbrock, generations, population_size, length, tournament_size, mutation_rate)
    print("Best Solution:", best_solution)
    print("Fitness of Best Solution:", fitness(best_solution))
    
    best_solution = genetic_algorithm(easom_function, generations, population_size, length, tournament_size, mutation_rate)
    print("Best Solution:", best_solution)
    print("Fitness of Best Solution:", fitness(best_solution))






