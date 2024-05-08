'''
Hybird Algorithm testing file 
It should get the datasets -- processe as required 
generate results 
'''
from utils import datasets
from HYB_PARSIMONY.hyb_parsimony import HybridParsimony


def test():
    """
    - Loads dataset information from a CSV file.
    - Processes each dataset by scaling features and executing the genetic algorithm.
    - Uses hyperparameter settings and genetic algorithm parameters from predefined settings.
    """
    hyperparam_template=[]
    hyperparam_template.append({'name':'C','lower_bound': 0.001, 'upper_bound': 1000})
    hyperparam_template.append({'name':'gamma','lower_bound': 0.001, 'upper_bound': 1000})
    # Loop through each dataset
    for (name,evaluator, num_features,original_df) in datasets(hyperparam_template,artificially_inflate=True):
        # HyperParameters and other settings for the hybrid algorithm
        population_size = 100
        elite_population_count = 20
        alpha=2.5
        beta=1.4
        gamma=0.5
        L=3
        max_iterations=20
        hybrid_parsimony = HybridParsimony(evaluator.get_objective_function(),num_features+len(hyperparam_template),population_size,max_iterations,
                                             alpha,beta,gamma,L,elite_population_count,len(hyperparam_template),evaluator)
                                           
        hybrid_parsimony.solve()                       
                                           
        print(f'{name} Dataset Loaded and Algorithm Executed')
        evaluator.print_results(original_df)
        