import numpy as np
from pyDOE2 import lhs


# Supporting functions go here as needed



####################
#  Hyperparameters
####################

# each individual is composed of the algorithm's training hyperparameters and
# the input features to the model
num_hyperparameters = 5
"""The number of training hyperparameters for the algorithm."""
num_features = 5
"""The number of input features to the model."""

# min and max values for the hyperparameters, can be individual for each
min_param = np.array([0]*10)  # Adjust this based on the actual range
"""
The minimum values for the hyperparameters. Each hyperparameter can have a
different minimum value.
"""
max_param = np.array([1]*10)  # Adjust this as well
"""
The maximum values for the hyperparameters. Each hyperparameter can have a
different maximum value.
"""

elite_size = 10

population_size = 50

num_generations = 30
"""The number of iterations to run the algorithm for."""

tournament_size = 5

mutation_rate = 0.01


############################
#  HYB-PARSIMONY algorithm
############################

# hyb-parsimony main algorithm
def hyb_parsimony():
    """
    This is the implementation of the HYB-PARSIMONY algorithm (Algorithm 3) from
    the paper.
    :return:
    """

    # 1. Initialization of positions X^0 using a random and uniformly
    # distributed Latin hypercube within the ranges of feasible values for each
    # input parameter.
    X = (lhs(num_hyperparameters + num_features, samples=population_size) *
         (max_param - min_param) + min_param)  # can split these as needed

    # 2. Initialization of velocities according to V^0 = (random_LHS(s, D) - X^0) / 2
    V = ((lhs(num_hyperparameters + num_features, samples=population_size) *
          (max_param - min_param) + min_param) - X) / 2

    # 3. Main loop
    for _ in range(num_generations):
        """"""
        # 4. Train each particle X_i(t) and validate with CV
        # TODO - need to figure out how to do this part
        #   Something like X[i] = train_and_validate(X[i], X, y)?

        # 5. Fitness evaluation J and complexity evaluation M_c of each particle
        # TODO - also need to figure this out

        # 6. Update the position of the best fitness value achieved by the ith
        # particle (X_hat_i), this particle's most parsimonious model (X_hat^p_i),
        # and the global best found so far (X_hat_hat).


        # (7. early stopping criterion - maybe ignore)
        # (8. early stopping return - maybe ignore)
        # (9. end early stopping check - maybe ignore)

        # 10. Generation of new neighborhoods if X_hat_hat did not improve


        # 11. Update each best position within a neighborhood (L_hat_i)


        # 12. Select elitist population P_e for reproduction


        # 13. Obtain a percentage (pcrossover) of worst individuals (P_w) to be
        # substituted with crossover


        # 14. Crossover P_e to substitute P_w with new individuals


        # 15. Update positions and velocities of P_e  (TODO only update P_e?)


        # 16. Mutation of a percentage of H (hyperparameters)


        # 17. Mutation of a percentage of F (features)


        # 18. Limitation of velocities and out-of-range positions (np.clip)


    # 19. end main loop
    # 20. Return X_hat_hat




###################
#  Run everything
###################

def main():
    """
    Main function, calls everything.
    """
    pass


if __name__ == '__main__':
    main()
