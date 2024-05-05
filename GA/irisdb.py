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
print(X)
print('Iris DataSet')
# Define hyperparameter ranges for the model
'''hyperparameter_ranges = [(0.01, 1.0), (2, 10)]

# Call the genetic algorithm function correctly
genetic_algorithm(data_features=X, target=y, hyperparameter_ranges=hyperparameter_ranges, generations=generations, population_size=population_size, 
                 elite_population_count=elite_population_count, mutation_rate=mutation_rate)'''

template:[{'lower_bound': 0, 'upper_bound': 1.0},{'lower_bound': 0, 'upper_bound': 1.0},
          {'lower_bound': 0, 'upper_bound': 1.0}, {'lower_bound': 0, 'upper_bound': 1.0},
           {name:'C','discreteValues': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]


        
       