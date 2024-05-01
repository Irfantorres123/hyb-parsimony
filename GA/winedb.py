import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from geneticParsimonyAlgo import genetic_algorithm  # Ensure this import is correctly set up

# Load the dataset, limiting to the first 2000 rows
df = pd.read_csv('wine.csv', nrows=2000)

# Assume the last column is the target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Preprocess data: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing (optional, based on your need for validation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# HyperParameters and other settings for the genetic algorithm
generations = 10
population_size = 10
elite_population_count = 5
mutation_rate = 0.01

# Define hyperparameter ranges for the model
hyperparameter_ranges = [(0.01, 1.0), (2, 10)]  # Adjust these based on the algorithm used in the genetic algorithm

# Call the genetic algorithm function
genetic_algorithm(data_features=X_train, target=y_train, hyperparameter_ranges=hyperparameter_ranges,
                  generations=generations, population_size=population_size,
                  elite_population_count=elite_population_count, mutation_rate=mutation_rate)

print('Wine Dataset Loaded and Algorithm Executed')
