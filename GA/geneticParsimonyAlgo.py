import random
import numpy as np
from pyDOE import lhs
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, log_loss

class Individual:
    """
    Represents an individual in the genetic algorithm population.

    Each individual comprises a set of binary-encoded features and associated hyperparameters.
    Fitness and complexity attributes are used to evaluate the performance and efficiency of the individual.

    Attributes:
    - features (np.array): Binary array indicating the presence (1) or absence (0) of each feature.
    - hyperparameters (dict): Dictionary of model hyperparameters.
    - fitness (float): Evaluation metric score of the individual, initialized to None.
    - complexity (float): Measure of model complexity, initialized to None.
    """
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
    """
    Initializes a population of individuals with binary-encoded features and scaled hyperparameters.
    
    This function generates a population where each individual is defined by a binary feature set
    and a set of hyperparameters. The features are initialized via Latin Hypercube Sampling to
    ensure even distribution, and hyperparameters are scaled based on provided ranges.
    
    Parameters:
    - num_individuals (int): Number of individuals in the population.
    - dataset_features (np.array): Dataset to determine the number of features.
    - hyperparameter_ranges (list of tuples): Min and max values for each hyperparameter.
    
    Returns:
    - population (list of Individual objects): Initialized population of individuals.
    """
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
    """
    Mutates binary features of an individual based on a given mutation rate.

    Parameters:
    - features (np.array): Binary array of features to mutate.
    - mutation_rate (float): Probability of each feature being mutated.

    Returns:
    - features (np.array): Mutated array of features.
    """
    # Vectorized mutation for binary features
    mutation_mask = np.random.rand(len(features)) < mutation_rate
    features[mutation_mask] = 1 - features[mutation_mask]
    return features

def mutation_hyperparameters(hyperparameters, mutation_rate, hyperparameter_ranges):
    """
    Mutates hyperparameters of an individual within given ranges based on a mutation rate.

    Parameters:
    - hyperparameters (list): List of hyperparameter values to mutate.
    - mutation_rate (float): Probability of each hyperparameter being mutated.
    - hyperparameter_ranges (list of tuples): Each tuple contains min and max values for a hyperparameter.

    Returns:
    - hyperparameters (list): List of mutated hyperparameter values.
    """
    '''Assumption that hyperparameter values are continuous values'''
    for key, value in enumerate(hyperparameters):
        if random.random() < mutation_rate:
            min_val, max_val = hyperparameter_ranges[key]
            range_val = max_val - min_val
            std_dev = 0.01 * range_val
            perturbation = np.random.normal(0, std_dev)
            new_value = value + perturbation
            hyperparameters[key] = np.clip(new_value, min_val, max_val)
    return hyperparameters

def mutation(individual, mutation_rate, hyperparameter_ranges):
    """
    Applies mutation to the features and hyperparameters of an individual.

    Parameters:
    - individual (Individual): The individual to mutate.
    - mutation_rate (float): Mutation rate to apply.
    - hyperparameter_ranges (list of tuples): Ranges for mutating hyperparameters.

    Returns:
    - individual (Individual): The mutated individual.
    """
    individual.features = mutation_features(individual.features, mutation_rate)
    individual.hyperparameters = mutation_hyperparameters(individual.hyperparameters, mutation_rate, hyperparameter_ranges)
    return individual


def genetic_algorithm(data_features, target, hyperparameter_ranges, generations=5, population_size=10, elite_population_count=5, 
                      mutation_rate=0.01):
    """
    Runs a genetic algorithm for feature selection and hyperparameter optimization.

    Initializes a population of individuals with selected features and hyperparameters,
    evolving over several generations through training, selection, crossover, and mutation.
    Includes early stopping if no improvement is seen for half the total generations.

    Parameters:
    - data_features (np.array): Dataset input features.
    - target (np.array): Dataset target variable.
    - hyperparameter_ranges (list of tuples): Min and max values for each hyperparameter.
    - generations (int): Total number of generations to run.
    - population_size (int): Number of individuals in each generation.
    - elite_population_count (int): Number of top individuals selected for reproduction.
    - mutation_rate (float): Probability of mutating an individual.

    Returns:
    - best_individual (Individual): The best individual from the last generation or upon early stopping.
    """
    try:
        population = initialize_population(population_size, data_features, hyperparameter_ranges)
        best_score = float('-inf')
        no_improvement_count = 0
        patience = generations/2
        
        for generation in range(generations):
            #print(f"Generation {generation+1}")
            for individual in population:
                individual.fitness, individual.complexity = train_and_validate(individual, data_features, target)
                #print(f"Individual with features {individual.features} has fitness: {individual.fitness:.2f}, complexity: {individual.complexity:.2f}")
            
            # Sort the population by fitness (descending) and by complexity (ascending) to break ties
            population.sort(key=lambda x: (-x.fitness, x.complexity))
            best_individual = population[0]
            print(f"Generation {generation + 1}:")
            print(f"Best Individual with features {best_individual.features} has fitness: {best_individual.fitness:.2f}, complexity: {best_individual.complexity:.2f}")
            print("------------------")
            
            # Early stopping check
            if best_individual.fitness > best_score:
                best_score = best_individual.fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"Stopping early after {generation+1} generations.")
                return best_individual

        # Elitism and reproduction
            elitist_population = population[:elite_population_count]
            new_population = []
            while len(new_population) < population_size:
                p1, p2 = random.sample(elitist_population, 2)
                offspring1, offspring2 = crossover(p1, p2)
                new_population.extend([offspring1, offspring2])
            
            # Mutation
            for i in range(len(new_population)):
                new_population[i] = mutation(new_population[i], mutation_rate, hyperparameter_ranges)
            
            population = new_population
        
        return population
    except Exception as e:
        print(f"Error during execution of algorithm: {e}")
        return None