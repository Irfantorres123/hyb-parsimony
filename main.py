####################
#  Runs everything
####################

import numpy as np
from hyb_parsimony import HybridParsimony
from benchmark_functions import easom


####################
#  Hyperparameters
####################

######################################
# Easom function hyperparameters:
# seed = 1234
# np.random.seed(seed)

D = 2

# lower_bound = -100.0
# upper_bound = 100.0

lower_bounds = np.array([-100.0] * D)
upper_bounds = np.array([100.0] * D)

num_particles = 50

alpha = 0.2
beta = 0.3

max_iterations = 100
num_trials = 30

# end Easom function hyperparameters
######################################

def f(x):
    """
    The cost function to minimize.

    :param x: input vector
    :return: the result of applying the function to the input vector
    """

    return easom(x, D)


def main():
    """
    Main function, calls everything.
    """

    # create the HYB-PARSIMONY algorithm
    hyb_parsimony = HybridParsimony(f, D, num_particles, alpha, beta,
                                    max_iterations,
                                    lower_bounds, upper_bounds)

    # run the algorithm
    # best_pos, best_f = hyb_parsimony.single_run()
    mean, std = hyb_parsimony.solve(num_trials)

    print(f"Mean optima found over {num_trials} trials: {mean}")
    print(f"Standard deviation: {std}")


if __name__ == '__main__':
    main()
