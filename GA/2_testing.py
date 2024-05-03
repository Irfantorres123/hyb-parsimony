import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from geneticParsimonyAlgo import genetic_algorithm  # Ensure this import is correctly set up

# Update the path as per your file structure
base_path = os.path.join('datasets') 

def load_and_process_dataset(filename, nrows=None):
    """
    Loads and preprocesses a dataset from a CSV file.

    Parameters:
    - filename (str): Name of the CSV file to load.
    - nrows (int, optional): Number of rows to read from the file.

    Returns:
    - X_scaled (np.array): Scaled feature data.
    - y (np.array): Target variable data.
    
    If the file is not found, returns None for both X_scaled and y.
    """
    file_path = os.path.join(base_path, filename)
    try:
        # Load the dataset
        df = pd.read_csv(file_path, nrows=nrows)
        # Assume the last column is the target variable
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Preprocess data: scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y
    
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None, None
 
def main():
    """
    Main function to execute the genetic algorithm on multiple datasets.
    
    - Loads dataset information from a CSV file.
    - Processes each dataset by scaling features and executing the genetic algorithm.
    - Uses hyperparameter settings and genetic algorithm parameters from predefined settings.
    """
    # Load dataset information
    db_path = os.path.join(base_path, 'res_basedata.csv')
    datasets_info =  pd.read_csv(db_path)

    # Loop through each dataset
    for index, row in datasets_info.iterrows():
        print(f"Processing dataset: {row['name_ds']}")
        
        # Assume CSV filename convention is "{name_ds}.csv", adjust if different
        dataset_filename = f"{row['name_ds']}.csv"
        X, y = load_and_process_dataset(dataset_filename, nrows=row['nrows'])

        # Split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # HyperParameters and other settings for the genetic algorithm
        generations = 10
        population_size = 10
        elite_population_count = 5
        mutation_rate = 0.01

        # Define hyperparameter ranges for the model, adjust based on needs
        hyperparameter_ranges = [(0.01, 1.0), (2, 10)]  # Example ranges

        # Call the genetic algorithm function
        genetic_algorithm(data_features=X_train, target=y_train, hyperparameter_ranges=hyperparameter_ranges,
                          generations=generations, population_size=population_size,
                          elite_population_count=elite_population_count, mutation_rate=mutation_rate)

        print(f'{row["name_ds"]} Dataset Loaded and Algorithm Executed')

if __name__ == '__main__':
    main()
