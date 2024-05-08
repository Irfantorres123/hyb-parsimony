# ---------------------------------------------------------------------------------
# Data Processing and Model Evaluation Framework
# ---------------------------------------------------------------------------------
# 
# Overview:
# This script is designed to streamline the process of loading, preprocessing, 
# and evaluating datasets for machine learning applications. It supports operations
# such as reading CSV files, optionally inflating datasets with synthetic features, 
# normalizing data, and setting up structured evaluations using Support Vector Machines (SVM).
#
# Features:
# - Load datasets from specified paths and manage partial dataset reads.
# - Artificially inflate datasets by adding randomly generated feature columns.
# - Standardize dataset features using sklearn's StandardScaler.
# - Generate configuration templates that combine feature range data with user-defined 
#   hyperparameters for model tuning.
# - Utilize an evaluation module for assessing the performance of SVM models on processed data.
#
# Dependencies:
# This script depends on several external libraries including pandas for data manipulation,
# NumPy for numerical operations, sklearn for data preprocessing and machine learning tasks,
# and a custom 'model_eval' module for setting up and running model evaluations.
#
# Usage:
# This script is intended to be used as a module in larger machine learning projects where
# dataset processing and model evaluation are required. Users can modify the base path,
# adjust hyperparameters, and select datasets for processing and evaluation through the 
# configuration settings defined within the script.
#
# ---------------------------------------------------------------------------------

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from model_eval import Evaluator
from sklearn import svm


base_path = os.path.join('datasets') 
def load_and_process_dataset(filename, nrows=None,artificially_inflate=False):
    """
    Load and process a dataset from a CSV file.

    Parameters:
    - filename (str): Name of the CSV file containing the dataset.
    - nrows (int): Number of rows to read from the dataset. If None, all rows are read.
    - artificially_inflate (bool): If True, artificially inflate the number of columns in the dataset by filling them with random values.

    """
    file_path = os.path.join(base_path, filename)
    try:
        original_df = pd.read_csv(file_path, nrows=nrows)
        # Assuming the last column is the target
        X = original_df.iloc[:, :-1].values
        y = original_df.iloc[:, -1].values
        num_cols=X.shape[1]
        num_new_cols = num_cols*0.25 # Add 25% more columns
        if artificially_inflate:
            # Artificially inflate the number of columns in the dataset by filling them with random values
            num_new_cols = int(num_new_cols)
            new_cols = np.random.rand(X.shape[0], num_new_cols)
            X = np.hstack((X, new_cols))
        columns = original_df.columns[:-1].tolist()+[f"rand{i}" for i in range(num_new_cols if artificially_inflate else 0)] + ["label"]
        original_df = pd.DataFrame(data=np.column_stack([X, y]), columns=columns)
        # Standard scaling of features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create a new DataFrame combining scaled features and target for evaluation purposes

        processed_df = pd.DataFrame(data=np.column_stack([X_scaled, y]), columns=columns)

        # Returning additional items to aid further processing
        column_bounds = [(processed_df[col].min(), processed_df[col].max()) for col in processed_df.columns[:-1]]
        print("Actual Rows in dataset-",nrows)
        num_features = len(processed_df.columns) - 1
        if num_features > 15: # Reducing dataset size if it has more than 15 features
            nrows = nrows//4
            processed_df = processed_df.sample(n=nrows, random_state=1)
        print("Rows processing-",nrows)
        return processed_df,original_df, column_bounds, num_features

    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None, None, None
    
def generate_template(column_bounds,hyperparam_template):
    """
    Combine feature bounds and hyperparameter template into a single configuration list.
    """
    template=[]
    for i in range(len(column_bounds)):
        template.append({'lower_bound': column_bounds[i][0], 'upper_bound': column_bounds[i][1]})
    template.extend(hyperparam_template)
    return template

def datasets(hyperparam_template,artificially_inflate=False):
    """
    Generator function to process each dataset described in a CSV file.
    """
    # Load dataset information
    db_path = os.path.join(base_path, 'res_basedata.csv')
    datasets_info =  pd.read_csv(db_path)
    for index, row in datasets_info.iterrows():
        name=row['name_ds']
        name=name.split("-")[0]
        print(f"Processing dataset: {name}")
        dataset_filename = f"{name}.csv" # Assume CSV filename convention is "{name_ds}.csv", adjust if different
        rows = row['nrows']
        
        df_data,original_df, column_bounds, num_features = load_and_process_dataset(dataset_filename, nrows=rows,artificially_inflate=artificially_inflate)

        template=generate_template(column_bounds,hyperparam_template)
        evaluator = Evaluator(template, num_features, svm.SVC, df_data)
        yield name,evaluator, num_features,original_df
