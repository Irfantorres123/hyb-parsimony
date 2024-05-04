####################
#  Runs everything
####################
import math

import numpy as np
# from hyb_parsimony import HybridParsimony
from hyb_parsimony import hyb_parsimony
from hyb_parsimony_modular import HybridParsimony
from benchmark_functions import easom
from benchmark_functions import rosenbrock
from tsb3342_PSO import particle_swarm_optimization


####################
#  Hyperparameters
####################

############################################################################
# Easom function hyperparameters:
# seed = 1234
# np.random.seed(seed)

D = 2
num_hyperparameters = 1

# lower_bound = -100.0
# upper_bound = 100.0

lower_bounds = np.array([-100.0] * D)
upper_bounds = np.array([100.0] * D)

num_particles = 50

alpha = 0.2
beta = 0.3

# L = 0.99
# L = 0.85  # inertia weight
L = 0.7  # inertia weight
# L = 0.5  # inertia weight
# L = 0.3  # inertia weight
# L = 0.001 # inertia weight

gamma = 0.3  # from Figure 3 in the paper

max_iterations = 100
# num_trials = 30
num_trials = 10
# num_trials = 5
# num_trials = 1

# end Easom function hyperparameters

elite_count = math.ceil(num_particles / 10)
############################################################################

def f(x):
    """
    The cost function to minimize.

    :param x: input vector
    :return: the result of applying the function to the input vector
    """

    return easom(x, D)
    # return rosenbrock(x, D)


def main():
    """
    Main function, calls everything.
    """

    # create the HYB-PARSIMONY algorithm
    # hyb_parsimony = HybridParsimony(f, D, num_particles, alpha, beta,
    #                                 max_iterations,
    #                                 lower_bounds, upper_bounds)
    # run the algorithm
    # best_pos, best_f = hyb_parsimony.solve()
    # print(f"Best position: {best_pos}, optima found: {best_f:.20f}")
    # mean, std = hyb_parsimony.solve(num_trials)
    # print(f"Mean optima found over {num_trials} trials: {mean:.20f}")
    # print(f"Standard deviation: {std}")



    # print(particle_swarm_optimization(f, D, num_particles, alpha, beta, max_iterations, lower_bounds, upper_bounds))
    # x_best, f_best = hyb_parsimony(f, D, num_particles, alpha, beta, L, gamma, max_iterations, elite_count, lower_bounds, upper_bounds)
    # print(f"Best position: {x_best}, optima found: {f_best:.20f}")


    hyb_parsimony_modular = HybridParsimony(f, D, num_particles, max_iterations,
                                            lower_bounds, upper_bounds, alpha,
                                            beta, gamma, L, elite_count,
                                            num_hyperparameters)
    # optima = hyb_parsimony_modular.solve()
    # print(f"Optima found: {optima}, value = {f(optima)}")
    x_best, f_best = hyb_parsimony_modular.solve()
    print(f"Best position: {x_best}, optima found: {f_best:.20f}")



if __name__ == '__main__':
    main()
