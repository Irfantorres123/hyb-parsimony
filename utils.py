import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        return X_scaled, y, df_data, column_bounds, len(df.columns)-1

    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None, None, None, None, None
    

def datasets():
    # Load dataset information
    db_path = os.path.join(base_path, 'res_basedata.csv')
    datasets_info =  pd.read_csv(db_path)
    for index, row in datasets_info.iterrows():
        name=row['name_ds']
        name=name.split("-")[0]
        print(f"Processing dataset: {name}")
        # Assume CSV filename convention is "{name_ds}.csv", adjust if different
        dataset_filename = f"{name}.csv"
        rows = row['nrows']
        
        X, y, df_data, column_bounds, num_features = load_and_process_dataset(dataset_filename, nrows=rows)
        print("Actual Rows in dataset-",rows)
        if num_features > 15: # Reducing dataset size if it has more than 15 features
            rows = 1000
            X = X[:rows]  
            y = y[:rows]
        print("Rows processing-",rows)

        yield df_data, column_bounds, num_features,name
