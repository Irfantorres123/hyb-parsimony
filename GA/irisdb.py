from geneticParsimonyAlgo import genetic_algorithm
from sklearn.datasets import load_iris

# HyperParameters and other settings
generations = 10
population_size = 10  # Not just 'population'
elite_population_count = 5
mutation_rate = 0.01
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
print('Iris DataSet')
# Define hyperparameter ranges for the model
hyperparameter_ranges = [(0.01, 1.0), (2, 10)]

# Call the genetic algorithm function correctly
genetic_algorithm(data_features=X, target=y, hyperparameter_ranges=hyperparameter_ranges, generations=generations, population_size=population_size, 
                 elite_population_count=elite_population_count, mutation_rate=mutation_rate)
