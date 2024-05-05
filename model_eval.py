from sklearn.svm import SVC
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd

# def FunctionModelWrapper(f):
#     """
#     Wrapper to convert a function to a model object with fit and score methods
#     """
#     class Model:
#         def __init__(self,**kwargs):
#             pass
#         def fit(self,X,y):
#             pass
#         def score(self,X,y):
#             return f(X)
#     return Model



class Evaluator:
    def __init__(self,template,num_features,model,dataset):
        """
        params:
        template: List of dictionaries containing info about that param including lower_bound, upper_bound,
                  name,type and discreteValues. discreteValues is a list of possible values for the param.
        num_features: Number of features in the dataset
        model: Model to be used for evaluation. can be any scikit learn model
        dataset: DataFrame. Dataset to be used for evaluation
        if discreteValues is provided, then name is needed however lower_bound and upper_bound are not needed.
        Example:
        template:[{'lower_bound': 0, 'upper_bound': 1.0},{name:'C','discreteValues': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        
        """
        self.template = template
        self.num_features = num_features
        self.num_hyperparameters = len(template) - num_features
        self.model =model
        self.dataset = dataset
        for tmp in template:
            if 'discreteValues' in tmp:
                assert 'name' in tmp
                tmp['discreteValues'] = np.array(tmp['discreteValues'])
            

    def clip(self,parameters:np.array):
        """
        Clip the parameters to their respective bounds and return them
        params:
        parameters: Numpy array containing the parameters to be clipped
        """
        for i in range(len(parameters)):
            template = self.template[i]
            if 'discreteValues' in template: # clip parameter to closest discrete value
                discreteValues = template['discreteValues']
                parameters[i] = discreteValues[np.argmin(np.abs(discreteValues - parameters[i]))]
            else:
                if parameters[i] < template['lower_bound']:
                    parameters[i] = template['lower_bound']
                elif parameters[i] > template['upper_bound']:
                    parameters[i] = template['upper_bound']
            if template['type']=='int':
                parameters[i] = round(parameters[i])
        return parameters
    
    def execute_agent(self,parameters:np.array):
        """
        Execute the model with the given parameters and return the score
        params:
        parameters: Numpy array containing the parameters to be used
        """
        hyperparameters = dict()
        features = []
        for i in range(len(parameters)):
            if i < self.num_features:
                features.append(parameters[i])
            else:
                hyperparameters[self.template[i]['name']] = parameters[i]
        
        model = self.model(**hyperparameters)
        features = np.array(features)
        features = self.clip(features)
        features = features > 0.5
        features = np.where(features)[0]
        features = self.dataset.columns[features]
        X = self.dataset[features].values
        y = self.dataset.iloc[:,-1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)

    def execute(self,agents:List[np.array],max_workers:int=4):
        """
        Execute the model with the given parameters and return the score
        params:
        agents: List of numpy arrays containing the parameters to be used
        max_workers: Number of threads to use for execution

        """
        thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        results = []
        for agent in agents:
            results.append(thread_pool.submit(self.execute_agent,agent))
        return [result.result() for result in results]
