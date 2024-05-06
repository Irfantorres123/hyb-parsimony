import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np

# Path where the geneticParsimonyAlgo module is located
module_path = '/Users/poonampawar/hyb-parsimony/GA'
sys.path.append(module_path)
from geneticParsimonyAlgo import genetic_algorithm 
from model_eval import Evaluator

# Update the path as per your file structure
base_path = os.path.join('datasets') 

def load_and_process_dataset(filename, nrows=None):
    file_path = os.path.join(base_path, filename)
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        # Assuming the last column is the target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Standard scaling of features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create a new DataFrame combining scaled features and target for evaluation purposes
        columns = df.columns[:-1].tolist() + ["label"]
        df_data = pd.DataFrame(data=np.column_stack([X_scaled, y]), columns=columns)

        # Returning additional items to aid further processing
        column_bounds = [(df[col].min(), df[col].max()) for col in df.columns[:-1]]
        return X_scaled, y, df_data, column_bounds, len(df.columns)

    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None, None, None, None, None
 
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
        rows = row['nrows']
    
        X, y, df_data, column_bounds, no_features = load_and_process_dataset(dataset_filename, nrows=rows)

        if X is not None and y is not None:
            print("Actual Rows in dataset-",rows)
            if no_features > 15:
                rows = 1000
                X = X[:rows]  # Slicing to first 1000 entries if more than that
                y = y[:rows]
            
            print("Rows processing-",rows)
            # HyperParameters and other settings for the genetic algorithm
            generations = 10
            population_size = 10
            elite_population_count = 5
            mutation_rate = 0.01
            hyperparameter_ranges = [(0.01, 1.0),[0.001,1]]
            
            template = []
            for i in range(len(column_bounds)):
                template.append({'lower_bound': column_bounds[i][0], 'upper_bound': column_bounds[i][1]})
            
            template.append({'name':'C','lower_bound': 0.01, 'upper_bound': 1})
            template.append({'name':'gamma','lower_bound': 0.001, 'upper_bound': 1})
            
            evaluator = Evaluator(template, no_features, svm.SVC, df_data)
            
            genetic_algorithm(data_features=X, target=y, 
                              hyperparameter_ranges=hyperparameter_ranges, 
                              generations=generations, population_size=population_size, 
                 elite_population_count=elite_population_count, 
                 mutation_rate=mutation_rate,evaluator=evaluator)


            print(f'{row["name_ds"]} Dataset Loaded and Algorithm Executed')
        else:
            print('Nothing to process')

if __name__ == '__main__':
    main()
