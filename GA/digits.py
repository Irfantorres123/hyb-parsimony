import genetic_algorithm from geneticParisomnyAlgo
from sklearn.datasets import load_digits

print("-------------")

print('Digits DataSet')
digits = load_digits()
X = digits.data
y = digits.target
# Example hyperparameter ranges for a logistic regression model
hyperparameter_ranges = [(0.01, 1.0), (2, 10)]  # C: regularization strength, max_depth: dummy example
# Call the genetic algorithm function correctly
genetic_algorithm(data_features=X, target=y, hyperparameter_ranges=hyperparameter_ranges, generations=generations, population_size=population_size, 
                 elite_population_count=elite_population_count, mutation_rate=mutation_rate)

