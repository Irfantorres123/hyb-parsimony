
from utils import datasets
# Path where the geneticParsimonyAlgo module is located
# module_path = '/Users/poonampawar/hyb-parsimony/GA'
# sys.path.append(module_path)
from GA.geneticParsimonyAlgo import genetic_algorithm 

def test():
    """
    - Loads dataset information from a CSV file.
    - Processes each dataset by scaling features and executing the genetic algorithm.
    - Uses hyperparameter settings and genetic algorithm parameters from predefined settings.
    """
    hyperparam_template=[]
    hyperparam_template.append({'name':'C','lower_bound': 0.01, 'upper_bound': 1})
    hyperparam_template.append({'name':'gamma','lower_bound': 0.001, 'upper_bound': 1})
    # Loop through each dataset
    for (name,evaluator, num_features,original_df) in datasets(hyperparam_template,artificially_inflate=True):
        # HyperParameters and other settings for the genetic algorithm
        generations = 10
        population_size = 10
        elite_population_count = 5
        mutation_rate = 0.01
        hyperparameter_ranges = [(0.01, 1.0),[0.001,1]]
        
        genetic_algorithm(num_features=num_features,
                          hyperparameter_ranges=hyperparameter_ranges, 
                          generations=generations, population_size=population_size, 
             elite_population_count=elite_population_count, 
             mutation_rate=mutation_rate,evaluator=evaluator)
        print(f'{name} Dataset Loaded and Algorithm Executed')
        evaluator.print_results(original_df)
        
    

if __name__ == '__main__':
    test()
