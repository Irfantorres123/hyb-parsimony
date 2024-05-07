'''
PSO Algorithm testing file 
It should get the datasets -- process as required 
generate results 
'''
from utils import datasets
from PSO.pso import ParticleSwarm


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
        # HyperParameters and other settings for the hybrid algorithm
        population_size = 10
        alpha=0.5
        beta=0.4
        
        max_iterations=10
        pso=ParticleSwarm(evaluator.get_objective_function(),max_iterations,population_size,evaluator,alpha,beta,num_features+len(hyperparam_template))
                                           
        pso.solve()                       
                                           
        print(f'{name} Dataset Loaded and Algorithm Executed')
        evaluator.print_results(original_df)
        