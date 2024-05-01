import random
import numpy as np
from pyDOE2 import lhs
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Constants
num_hyperparameters = 2  # Let's assume these are n_estimators and max_depth for RandomForest
min_param = np.array([10, 1])  # Minimum values for n_estimators and max_depth
max_param = np.array([100, 10])  # Maximum values for n_estimators and max_depth
population_size = 10
generations = 10
elite_size = 2
tournament_size = 5
mutation_rate = 0.05
threshold = 1e-6

def create_individual():
    # Generate hyperparameters using Latin hypercube sampling
    hyperparameters = lhs(len(min_param), samples=1) * (max_param - min_param) + min_param
    return hyperparameters.flatten()

def train_and_validate(individual):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=int(individual[0]), max_depth=int(individual[1]), random_state=42)
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_test, y_test, cv=5).mean()
    return score

def fitness_and_complexity(individual):
    fitness = train_and_validate(individual)
    complexity = -individual[1]  # Assuming lower depth is less complex
    return fitness, complexity

def sort_and_promote(population):
    # Sort primarily by fitness, then by complexity within similar fitness levels
    return sorted(population, key=lambda x: (-fitness_and_complexity(x)[0], fitness_and_complexity(x)[1]))

def mutate(individual):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(min_param[idx], max_param[idx])
    return individual

def genetic_algorithm():
    population = [create_individual() for _ in range(population_size)]
    best_score = -np.inf
    best_individual = None

    for generation in range(generations):
        population = sort_and_promote(population)
        next_generation = population[:elite_size]  # Elitism

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1 = mutate(parent1.copy())
            child2 = mutate(parent2.copy())
            next_generation.extend([child1, child2])

        population = next_generation
        current_best = population[0]
        current_score = fitness_and_complexity(current_best)[0]

        if current_score > best_score:
            best_score = current_score
            best_individual = current_best

        print(f"Generation {generation}: Best Fitness = {best_score}")

        # Early stopping check (optional)
        '''if generation > 0 and best_score >= threshold:
            print("Early stopping triggered.")
            break'''

    return best_individual

if __name__ == "__main__":
    best_solution = genetic_algorithm()
    print("Best Solution:", best_solution)
    print("Fitness of Best Solution:", fitness_and_complexity(best_solution)[0])
