import random
import numpy as np
from pyDOE import lhs
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, log_loss

class Individual:
    def __init__(self, features, hyperparameters):
        self.features = features
        self.hyperparameters = hyperparameters
        self.fitness = None  # Initialize fitness
        self.complexity = None  # Initialize complexity

    
def initialize_features(num_individuals, num_features):
    """ Generate Latin Hypercube Samples for features.
        Features are binary, indicating whether a feature is included (1) or excluded (0). """
    lhs_features = lhs(num_features, samples=num_individuals, criterion='corr')
    features = np.round(lhs_features)
    return features.astype(int)

def initialize_hyperparameters(num_individuals, hyperparameter_ranges):
    """ Generate Latin Hypercube Samples for hyperparameters, scaled to the specified ranges.
        Each hyperparameter is described as follows:
        - Index 0: 'C' (regularization strength) for logistic regression,
        - Index 1: 'max_depth' (maximum depth of the tree) for decision trees. """
    num_hyperparameters = len(hyperparameter_ranges)
    lhs_hyperparameters = lhs(num_hyperparameters, samples=num_individuals, criterion='corr')
    hyperparameters = []
    for i in range(num_individuals):
        hp = {}
        for j, (min_val, max_val) in enumerate(hyperparameter_ranges):
            hp[j] = min_val + lhs_hyperparameters[i, j] * (max_val - min_val)
        hyperparameters.append(hp)
    return hyperparameters

def initialize_population(num_individuals, dataset_features, hyperparameter_ranges):
    num_features = dataset_features.shape[1]  # Determine the number of features from the dataset
    features = initialize_features(num_individuals, num_features)
    hyperparameters = initialize_hyperparameters(num_individuals, hyperparameter_ranges)
    population = []
    for i in range(num_individuals):
        individual = Individual(features[i], hyperparameters[i])
        population.append(individual)
    return population

def evaluate_fitness(y_true, y_pred, y_proba, method='f1', average='macro'):
    if method == 'f1':
        return f1_score(y_true, y_pred, average=average)
    elif method == 'auc':
        # AUC can be calculated per class and then averaged if needed
        return roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
    elif method == 'logloss':
        return -log_loss(y_true, y_proba)  # For log-loss, 'average' is not used
    else:
        raise ValueError("Unknown fitness evaluation method")

def evaluate_complexity(individual, hyperparameters):
    """ Calculate complexity based on number of features and parameter values. Irfan's Code"""
    num_features_used = np.sum(individual.features == 1)
    c_penalty = hyperparameters[0]
    gamma_value = hyperparameters[1]
    
    # Example complexity formula
    feature_complexity = 1 / num_features_used if num_features_used > 0 else float('inf')
    parameter_complexity = c_penalty * gamma_value  # Simplistic interaction
    return feature_complexity + parameter_complexity

def train_and_validate(individual, X, y, fitness_method='f1'):
    # Select features based on individual's feature mask
    X_selected = X[:, individual.features == 1]

    # Check if selected features array is empty
    if X_selected.size == 0:
        print("No features selected, skipping this individual.")
        return 0, float('inf')  # Return a low score or handle as appropriate

    # Create an SVM model. Assuming hyperparameters are in the order [C, gamma]
    model = svm.SVC(C=individual.hyperparameters[0], gamma=individual.hyperparameters[1], probability=True)

    # Perform cross-validation and return the average accuracy
    try:
        scores = cross_val_score(model, X_selected, y, cv=5)  # Using 5-fold cross-validation
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)
        y_proba = model.predict_proba(X_selected)[:, 1]
        fitness = evaluate_fitness(y, y_pred, y_proba, method=fitness_method)
    except Exception as e:
        print(f"Error during model training: {e}")
        return 0, float('inf') # Return a low score or handle as appropriate
    
    complexity = evaluate_complexity(individual, (individual.hyperparameters[0], individual.hyperparameters[1]))
    return fitness, complexity

def crossover(parent1, parent2):
    '''
    Perform crossover between two parent individuals to generate two offsprings.
    This method perform single point crossover for both the features and hype parameters.
    A random crossover point is selected for the features arrays and for the hyperparameters array.
    
    Parameters:
    - parent1 (Individual): The first parent 
    - parent2 (Individual): The second parent
    
    Returns:
    - offspring1: The first offspring
    - offspring2: The second offspring
    '''
    #select crossover point 
    features_len = len(parent1.features)
    crossover_point = random.randint(0, features_len-1)
    
    #Features crossover
    offspring1_features = np.concatenate([parent1.features[:crossover_point], parent2.features[crossover_point:]])
    offspring2_features = np.concatenate([parent2.features[:crossover_point], parent1.features[crossover_point:]])
    
    hyperparam_keys = list(parent1.hyperparameters.keys())
    hyperparam_len = len(parent1.hyperparameters)
    crossover_point = random.randint(1, hyperparam_len-1)
    
    offspring1_hyperparameters = {}
    offspring2_hyperparameters = {}
    
    for key in hyperparam_keys[:crossover_point]:
        offspring1_hyperparameters[key] = parent1.hyperparameters[key]
        offspring2_hyperparameters[key] = parent2.hyperparameters[key]
        
    for key in hyperparam_keys[crossover_point:]:
        offspring1_hyperparameters[key] = parent2.hyperparameters[key]
        offspring2_hyperparameters[key] = parent1.hyperparameters[key]
    
    
    offspring1= Individual(offspring1_features, offspring1_hyperparameters)
    offspring2= Individual(offspring2_features, offspring2_hyperparameters)
    
    return offspring1, offspring2
    

def mutation_features(features, mutation_rate):
    '''Assumption that features are binary encoded'''
    for i in range(len(features)):
        if random.random() < mutation_rate:
            if features[i] == 0:
                features[i] = 1
            else:
                features[i] = 0
    
    return features

def mutation_hyperparameters(hyperparameters, mutation_rate):
    '''Assumption that hyperparameter values are continuous values'''
    for key in hyperparameters.keys():
        if random.random() < mutation_rate:
            perturbation = normal_distribution(mean = 0, std=0.01*hyperparameters[key])
            hyperparameters[key]+=perturbation
    
    return hyperparameters
     
def mutation(Individual, mutation_rate):
    Individual.features = mutation_features(Individual.features, mutation_rate)
    Individual.hyperparameters = mutation_features(Individual.hyperparameters, mutation_rate)
    return Individual

def genetic_algorithm(data_features, target, hyperparameter_ranges, generations=5, population_size=10, elite_population_count=5, 
                      mutation_rate=0.01):
    population = initialize_population(population_size, data_features, hyperparameter_ranges)
    print("------------------")
    for generation in range(generations):
        print(f"Generation {generation+1}")
        for individual in population:
            individual.fitness, individual.complexity = train_and_validate(individual, data_features, target)
            #print(f"Individual with features {individual.features} has fitness: {individual.fitness:.2f}, complexity: {individual.complexity:.2f}")
        
        # Sort the population by fitness (descending) and by complexity (ascending) to break ties
        population.sort(key=lambda x: (-x.fitness, x.complexity))
        best_individual = population[0]
        print(f"Generation {generation + 1}:")
        print(f"Best Individual with features {best_individual.features} has fitness: {best_individual.fitness:.2f}, complexity: {best_individual.complexity:.2f}")
        print("------------------")
        
        # To do: Promote individuals with best 𝑀𝑐 between those with similar 𝐽
        # To do: early stopping
        elitist_population = population[: elite_population_count]
        # Cross over 𝑃𝑒 to create a new generation 𝐗𝑡+1
        new_population = []
        cnt = 0
        while cnt < elite_population_count:
            p1 = random.randint(0, elite_population_count-1)
            p2 = random.randint(0, elite_population_count-1)
            offspring1, offspring2 = crossover(elitist_population[p1], elitist_population[p2])
            new_population.append(offspring1)
            new_population.append(offspring2)
            cnt += 1
            
        for i in range(len(new_population)):
            ind = mutation(new_population[i], mutation_rate)
            new_population[i] = ind
    
        population = new_population
            
    return population

    
    
# HyperParameters and other settings
generations = 10
population_size = 10  # Not just 'population'
elite_population_count = 5
mutation_rate = 0.01
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
'''print('Iris DataSet')
# Define hyperparameter ranges for the model
hyperparameter_ranges = [(0.01, 1.0), (2, 10)] 

# Call the genetic algorithm function correctly
genetic_algorithm(data_features=X, target=y, hyperparameter_ranges=hyperparameter_ranges, generations=generations, population_size=population_size, 
                 elite_population_count=elite_population_count, mutation_rate=mutation_rate)'''

    
print('Digits DataSet')
digits = load_digits()
X = digits.data
y = digits.target
# Example hyperparameter ranges for a logistic regression model
hyperparameter_ranges = [(0.01, 1.0), (2, 10)]  # C: regularization strength, max_depth: dummy example
# Call the genetic algorithm function correctly
genetic_algorithm(data_features=X, target=y, hyperparameter_ranges=hyperparameter_ranges, generations=generations, population_size=population_size, 
                 elite_population_count=elite_population_count, mutation_rate=mutation_rate)


