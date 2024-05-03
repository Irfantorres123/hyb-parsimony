import os
import openml
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# Update the path as per your file structure
# Define the path to store the datasets CSV file
base_path = os.path.join('datasets')  # Adjust the path to navigate up one level and then to 'dataset'
print(base_path)
csv_path = os.path.join(base_path, 'res_datasets.csv')
print(csv_path)
log_path = os.path.join(base_path, 'downloaded_datasets_log.csv')
print(log_path)

# Ensure the dataset directory exists
if not os.path.exists(base_path):
    os.makedirs(base_path)

def res_datasets_exists(path):
    # Check if the datasets CSV file exists at the specified path
    try:
        pd.read_csv(path)
        return True
    except FileNotFoundError:
        return False

if not res_datasets_exists(csv_path):
    datasets_openml = openml.datasets.list_datasets(output_format="dataframe")
    print(f"Total datasets available: {datasets_openml.shape}")

    # 10 Datasets are loaded here, update as per requirements
    selec_datasets = datasets_openml.query(
        'NumberOfInstances > 1000 and NumberOfInstances < 30000 and '
        'NumberOfFeatures > 10 and NumberOfFeatures < 100'
    ).head(10)
    print(f"Selected datasets for download: {selec_datasets.shape}")

    res_datasets = []
    for index, dataset_info in tqdm(selec_datasets.iterrows(), total=selec_datasets.shape[0]):
        dataset_id = dataset_info['did']
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
            if y is not None:
                res_datasets.append({
                    'id_ds': dataset.dataset_id,
                    'name_ds': dataset_info['name'],
                    'nrows': X.shape[0],
                    'len_y': len(y),
                    'type_y': y.dtypes,
                    'num_classes': len(np.unique(y)),
                    'target_name': dataset.default_target_attribute
                })
        except Exception as e:
            print(f"Failed to download dataset ID {dataset_id}: {e}")
            continue

    res_datasets = pd.DataFrame(res_datasets)
    res_datasets.to_csv(csv_path, index=False)
else:
    res_datasets = pd.read_csv(csv_path)

print(f"Datasets loaded: {res_datasets.shape}")


res_selec = res_datasets.query(
    'type_y != "object" and type_y != "Sparse[float64, 0]" and '
    'len_y > 2000 and type_y != "int64" and nrows < 40000'
)
res_selec = res_selec.drop_duplicates()
print(f"After removing duplicates: {res_selec.shape}")

selec_names = []
pos_selec = []
for nrow, name_ds in enumerate(res_selec.name_ds.values):
    three = name_ds[:3]
    if three not in selec_names:
        selec_names.append(three)
        pos_selec.append(nrow)

res_selec = res_selec.iloc[pos_selec]

# 10 Datasets are loaded here, update as per requirements
# Ensure that we only have exactly 10 datasets, if there are more, select the first 10
if len(res_selec) > 10:
    res_selec = res_selec.head(10)

print(f"Final selection: {res_selec.shape}")
print(res_selec.head())

# Save the final selection to a CSV file
res_selec.to_csv(log_path, index=False)

warnings.filterwarnings("ignore")
# Inputs
# array(['bool', 'category', 'float64', 'int64', 'object', 'uint8'],

# Outputs
# array(['bool', 'category', 'float64', 'uint8'], dtype='<U8')

scaler = StandardScaler()
label_enc = LabelEncoder()
res_basedata = []
for nrow, row in tqdm(res_selec.iterrows(), total=len(res_selec)):
    try:
        # Get dataset by name
        dataset = openml.datasets.get_dataset(row['name_ds'])
        # Get the data itself as a dataframe (or otherwise)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        assert y.isna().sum()==0
        
        # Transform Inputs
        # ----------------
        X_prim = []
        for namecol in X.columns:
            # Convert numeric to float64
            if X[namecol].dtypes==np.uint8 or X[namecol].dtypes==np.int64 or X[namecol].dtypes==np.float64 or X[namecol].dtypes==bool:
                X[namecol] = X[namecol].astype(np.float64)
                X[namecol] = X[namecol].fillna(X[namecol].mean())
                X[namecol] = scaler.fit_transform(X[namecol].values.reshape(-1,1)).flatten()
                X_prim.append(X[namecol])
            if str(X[namecol].dtypes)=='category' or str(X[namecol].dtypes)=='object':
                X_prim.append(pd.get_dummies(X[namecol], prefix=namecol).astype(np.float32))
        X_prim = pd.concat(X_prim, axis=1)
        
        # Transform Target
        # --------------
        if row['type_y'] == 'float64':
            y.fillna(y.mean(), inplace=True)
            y_values = y.values.flatten()
            y_values = scaler.fit_transform(y_values.reshape(-1,1)).flatten()
            type_prob = 'regression'
        if row['type_y'] == 'uint8':
            y = y.astype(np.int32)
            y_values = y.values.flatten()
            type_prob = 'binary' if row['num_classes']==2 else 'multiclass'
            
        if row['type_y'] == 'bool':
            y = y.astype(np.int32)
            y_values = y.values.flatten()
            type_prob = 'binary' if row['num_classes']==2 else 'multiclass'
        if row['type_y'] == 'category':
            y_values = label_enc.fit_transform(y)
            type_prob = 'binary' if row['num_classes']==2 else 'multiclass'
        
        name_file = row['name_ds'].replace('-','_').split('_')[0]
        print(name_file)
        
        # Include more 
        row['name_file'] = name_file + '.csv'
        row['type_prob'] = type_prob
        row['NFs'] = X_prim.shape[1]
        X_prim['target_end'] = y_values
        res_basedata.append(row)
        
        if type_prob == 'regression':
            print(row['name_ds'], X.shape, X_prim.shape, type_prob, X_prim['target_end'].mean(), X_prim['target_end'].std())
        if type_prob == 'binary' or type_prob == 'multiclass':
            print(row['name_ds'], X.shape, X_prim.shape, type_prob, X_prim['target_end'].nunique())
        
        dataset_filename = os.path.join(base_path, row['name_file'])
        X_prim.to_csv(dataset_filename, index=False)
    except Exception as e:
        print(f"Failed processing {row['name_ds']}: {e}")  # Explicitly log errors

final_csv_path = os.path.join(base_path, 'res_basedata.csv')
res_basedata = pd.DataFrame(res_basedata)
res_basedata.to_csv(final_csv_path, index=False)  # Saving the summary CSV
