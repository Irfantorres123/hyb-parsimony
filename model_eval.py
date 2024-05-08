from sklearn.svm import SVC
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd


class Evaluator:
    def __init__(self, template=None, num_features=None, model=None, dataset=None):
        """
        Initializes the evaluator with optional parameters. If parameters are not provided
        during initialization, they must be set later before evaluation functions are used.
        
        Parameters:
        - template (list): Specifications for each parameter, including bounds or discrete values.
        - num_features (int): Number of features in the dataset.
        - model (object): Machine learning model that follows scikit-learn's interface.
        - dataset (DataFrame): The dataset used for training and evaluation.
        """
        self.template = template
        self.num_features = num_features
        self.num_hyperparameters = len(template) - num_features if template and num_features else None
        self.model = model
        self.dataset = dataset
        self.best_agent = None
        self.best_agent_score = None
        self.best_agent_accuracy = None
        self.validate_template()
        
    def validate_template(self):
        if self.template:
            for tmp in self.template:
                if 'discreteValues' in tmp:
                    assert 'name' in tmp, "If 'discreteValues' are provided, 'name' must also be specified."
                    tmp['discreteValues'] = np.array(tmp['discreteValues'])
    
    def set_template(self, template):
        self.template = template
        self.num_hyperparameters = len(template) - self.num_features if self.num_features else None
        self.validate_template()

    def set_num_features(self, num_features):
        self.num_features = num_features
        self.num_hyperparameters = len(self.template) - num_features if self.template else None

    def set_model(self, model):
        self.model = model
        
    def set_dataset(self, dataset):
        self.dataset = dataset
        
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
            if template.get('type')=='int':
                parameters[i] = round(parameters[i])
        return parameters

    def get_fitness(self,score,num_features,total_features):
        """
        Calculate the fitness of the model based on the score and number of features. Fitness is to be maximized.
        Score is supposed to be maximized and number of features is supposed to be minimized.
        params:
        score: Score of the model
        num_features: Number of features used in the model
        """
        ratio = num_features/total_features
        return score - 0.3*ratio

    def execute_agent(self,parameters:np.array,dataset):
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
        features = dataset.columns[features]
        if len(features) == 0:
            return 0,0,0
        X = dataset[features].values
        y = dataset.iloc[:,-1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train,y_train)
        score,num_features=model.score(X_test,y_test), len(features)
        return (self.get_fitness(score,num_features,self.num_features),score,num_features)

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
            results.append(thread_pool.submit(self.execute_agent,agent,self.dataset))
        final_results= [result.result() for result in results]
        for i in range(len(agents)):
            if self.best_agent is None or final_results[i][0] > self.best_agent_score:
                self.best_agent = agents[i]
                self.best_agent_score = final_results[i][0]
                self.best_agent_accuracy = final_results[i][1]
        return final_results

    def get_objective_function(self):
        """
        Return the objective function to be used for optimization
        """
        def objective_function(agents):
            return self.execute(agents)
        return objective_function

    def print_results(self,original_df:pd.DataFrame):
        """
        Print the results of the evaluation
        """
        print(f"Model: {self.model}")
        print(f"Best Agent fitness: {self.best_agent_score}")
        print(f"Best Agent accuracy: {self.best_agent_accuracy}")
        columns=self.dataset.columns
        chosen_feature_names = [columns[i] for i in range(self.num_features) if self.best_agent[i] > 0.5]
        print(f"Chosen Features: {chosen_feature_names}")
        best_hyperparameters = {self.template[i]['name']:self.best_agent[i] for i in range(self.num_features,len(self.best_agent))}
        print(f"Best Hyperparameters: {best_hyperparameters}")
        print("Training new model with best hyperparameters and features and complete data")
        
        fitness,accuracy,num_f=self.execute_agent(self.best_agent,original_df)
        actual_columns=len([1 for i in range(self.num_features) if "rand" not in columns[i]])
        artificial_columns=len(columns)-actual_columns
        actual_columns_chosen=len([1 for i in range(len(chosen_feature_names)) if "rand" not in chosen_feature_names[i]])
        artificial_columns_chosen=len(chosen_feature_names)-actual_columns_chosen
        print(f"Final Model accuracy: {accuracy}")
        print(f"Number of actual columns: {actual_columns}")
        print(f"Number of artificial columns: {artificial_columns}")
        print(f"Fitness: {fitness}")
        print("Percentage of actual features chosen:",actual_columns_chosen/actual_columns)
        print("Percentage of artificial features chosen:",artificial_columns_chosen/artificial_columns)
        print()
        print()
