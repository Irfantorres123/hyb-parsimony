import numpy as np
from pyDOE2 import lhs
import math


class HybridParsimony:
    """
    Implements the HYB-PARSIMONY algorithm. This algorithm is based on particle
    swarm optimization, but makes use of mutation and crossover from genetic
    algorithms.
    """

    def __init__(self, f, D, num_particles, alpha, beta, max_iterations,
                 lower_bounds, upper_bounds):
        """
        Initializes an instance of the HYB-PARSIMONY algorithm using the given
        parameters.

        :param f: the objective function to optimize
        :param D: the number of dimensions to optimize this function in
        :param num_particles: the population size
        :param alpha: controls the influence of the global best value
        :param beta: controls the influence of each particle's personal best value
        :param max_iterations: the number of iterations to run the algorithm for
        :param lower_bounds: the minimum values for each dimension
        :param upper_bounds: the maximum values for each dimension
        """

        # initialize the parameters
        self.f = f
        self.D = D
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # create the particles
        self.particles = [Particle(self.f, self.D, self.alpha, self.beta, self.lower_bounds, self.upper_bounds)
                          for _ in range(self.num_particles)]

        # initialize the current global best value among the particles
        self.global_best_pos = self.particles[0].best_pos
        self.global_best_f = self.particles[0].best_f
        # self.global_best_pos = self.particles[0].pos
        # self.global_best_f = self.particles[0].current_f
        for particle in self.particles:
            self.update_global_best(particle)


    def update_global_best(self, particle):
        """
        Updates the global best position if the given particle's best position
        is better.

        :param particle: the particle to check
        :return: None
        """

        if particle.best_f < self.global_best_f:
            self.global_best_pos = particle.best_pos
            self.global_best_f = particle.best_f
        #
        # if particle.current_f < self.global_best_f:
        #     self.global_best_pos = particle.pos
        #     self.global_best_f = particle.current_f


    def single_run(self):
        """
        Runs a single instance of the HYB-PARSIMONY algorithm to optimize the
        given objective function over the given dimensions using the given
        number of iterations.

        :return: The best position and f value found.
        """

        for _ in range(self.max_iterations):  # run for this number of iterations
            # Idea: update each particle, and then update the global best
            for particle in self.particles:
                particle.update(self.global_best_pos)
                self.update_global_best(particle)

            # for particle in self.particles:
            #     self.update_global_best(particle)

        return self.global_best_pos, self.global_best_f


    def solve(self, num_trials=1):
        """
        Runs the HYB-PARSIMONY algorithm for the given number of trials.

        :return: the mean and standard deviation of the best f values found.
        """

        # found_optima = []  # for the f(x) from each trial
        #
        # for _ in range(num_trials):
        #     # For each trial, we find a solution and add it to the list
        #     (x_best, f_best) = self.single_run()
        #     found_optima.append(f_best)
        #     self.reset()
        #
        # # calculate statistics
        # mean = np.mean(found_optima)
        # std = np.std(found_optima)
        # return mean, std

        return self.single_run()


    def reset(self):
        """
        Resets this algorithm's particles to their initial random states.

        :return: None
        """

        for particle in self.particles:
            particle.pos = (lhs(self.D, samples=1) * (self.upper_bounds - self.lower_bounds) + self.lower_bounds).flatten()
            particle.velocity = (lhs(self.D, samples=1) * (self.upper_bounds - self.lower_bounds) + self.lower_bounds).flatten()
            particle.current_f = self.f(particle.pos)
            particle.best_pos = particle.pos
            particle.best_f = particle.current_f

        self.global_best_pos = self.particles[0].best_pos
        self.global_best_f = self.particles[0].best_f


        # self.particles = [Particle(self.f, self.D, self.alpha, self.beta,
        #                            self.lower_bounds, self.upper_bounds)
        #                   for _ in range(self.num_particles)]
        # self.global_best_pos = self.particles[0].pos
        # self.global_best_f = self.particles[0].current_f
        # for particle in self.particles:
        #     self.update_global_best(particle)


class Particle:
    """
    Represents a particle in the HYB-PARSIMONY algorithm. Each particle has a
    position, velocity, and personal best value. Additionally, each particle has
    an objective function to optimize, plus bounds for its position.
    """

    def __init__(self, f, D, alpha, beta, lower_bounds, upper_bounds):
        """
        Initializes a single particle.

        :param f: the objective function to optimize
        :param D: the number of dimensions to optimize this function in
        :param alpha: controls the influence of the global best value
        :param beta: controls the influence of each particle's personal best value
        :param lower_bounds: the minimum values for each dimension
        :param upper_bounds: the maximum values for each dimension
        """

        # initialize the parameters
        self.f = f
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # initialize the position and velocity from a latin hypercube sample
        # also make sure that the arrays are flattened, i.e. [x, y] instead of [[x, y]]
        self.pos = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()
        self.velocity = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()

        # self.current_f = f(self.pos)

        # initialize the personal best values
        self.best_pos = self.pos
        self.best_f = f(self.best_pos)

        # TODO: keep track of the most parsimonious model within tolerance
        # self.best_complexity_pos = self.pos
        # self.best_complexity_f = f(self.best_complexity_pos)


    def update(self, global_best_pos):
        """
        Updates this particle's velocity and position.

        :param global_best_pos: the global best position found so far
        :return: None
        """

        epsilon1 = np.random.uniform()
        epsilon2 = np.random.uniform()

        # first update velocity, then the position, using this new velocity
        # self.velocity += self.alpha * epsilon1 * (global_best_pos - self.pos) + \
        #                  self.beta * epsilon2 * (self.best_pos - self.pos)
        self.velocity = 0.7 * self.velocity + self.alpha * epsilon1 * (global_best_pos - self.pos) + \
                         self.beta * epsilon2 * (self.best_pos - self.pos)

        self.pos += self.velocity
        # stay within bounds
        self.pos = np.clip(self.pos, a_min=self.lower_bounds, a_max=self.upper_bounds)

        # then update this particle's best, as needed
        f_pos = self.f(self.pos)
        # self.current_f = f_pos
        if f_pos < self.best_f:
        # if self.current_f < self.best_f:
        #     self.best_f = self.current_f
            self.best_f = f_pos
            self.best_pos = self.pos





####################
#  Hyperparameters
####################
#
# # each individual is composed of the algorithm's training hyperparameters and
# # the input features to the model
# num_hyperparameters = 5
# """The number of training hyperparameters for the algorithm."""
# num_features = 5
# """The number of input features to the model."""
#
# # min and max values for the hyperparameters, can be individual for each
# lower_bounds = np.array([0]*10)  # Adjust this based on the actual range
# """
# The minimum values for the hyperparameters. Each hyperparameter can have a
# different minimum value.
# """
# upper_bounds = np.array([1]*10)  # Adjust this as well
# """
# The maximum values for the hyperparameters. Each hyperparameter can have a
# different maximum value.
# """
#
# elite_size = 10
#
# population_size = 50
#
# num_generations = 30
# """The number of iterations to run the algorithm for."""
#
# tournament_size = 5
#
# mutation_rate = 0.01
#
#
# # ############################
# #  HYB-PARSIMONY algorithm
# ############################
#
# # hyb-parsimony main algorithm
# def hyb_parsimony():
#     """
#     This is the implementation of the HYB-PARSIMONY algorithm (Algorithm 3) from
#     the paper.
#     :return:
#     """
#
#     # 1. Initialization of positions X^0 using a random and uniformly
#     # distributed Latin hypercube within the ranges of feasible values for each
#     # input parameter.
#     X = (lhs(num_hyperparameters + num_features, samples=population_size) *
#          (upper_bounds - lower_bounds) + lower_bounds)  # can split these as needed
#
#     # 2. Initialization of velocities according to V^0 = (random_LHS(s, D) - X^0) / 2
#     V = ((lhs(num_hyperparameters + num_features, samples=population_size) *
#           (upper_bounds - lower_bounds) + lower_bounds) - X) / 2
#
#     # 3. Main loop
#     for _ in range(num_generations):
#         """"""
#         # 4. Train each particle X_i(t) and validate with CV
#         # TODO - need to figure out how to do this part
#         #   Something like X[i] = train_and_validate(X[i], X, y)?
#
#         # 5. Fitness evaluation J and complexity evaluation M_c of each particle
#         # TODO - also need to figure this out
#
#         # 6. Update the position of the best fitness value achieved by the ith
#         # particle (X_hat_i), this particle's most parsimonious model (X_hat^p_i),
#         # and the global best found so far (X_hat_hat).
#
#
#         # (7. early stopping criterion - maybe ignore)
#         # (8. early stopping return - maybe ignore)
#         # (9. end early stopping check - maybe ignore)
#
#         # 10. Generation of new neighborhoods if X_hat_hat did not improve
#
#
#         # 11. Update each best position within a neighborhood (L_hat_i)
#
#
#         # 12. Select elitist population P_e for reproduction
#
#
#         # 13. Obtain a percentage (pcrossover) of worst individuals (P_w) to be
#         # substituted with crossover
#
#
#         # 14. Crossover P_e to substitute P_w with new individuals
#
#
#         # 15. Update positions and velocities of P_e  (TODO only update P_e?)
#
#
#         # 16. Mutation of a percentage of H (hyperparameters)
#
#
#         # 17. Mutation of a percentage of F (features)
#
#
#         # 18. Limitation of velocities and out-of-range positions (np.clip)
#
#
#     # 19. end main loop
#     # 20. Return X_hat_hat


# class Particle:
#     def __init__(self, features, hyperparameters):
#         self.features = features
#         self.hyperparameters = hyperparameters
#         self.fitness = None
#         self.complexity = None


def hyb_parsimony(f, D, n, alpha, beta, L, gamma, max_iterations, elite_count,
                  lower_bounds, upper_bounds):
    """
    Implements the hybrid parsimony algorithm from Algorithm 3 of
    https://doi.org/10.1016/j.neucom.2023.126840

    :param f: the objective function to optimize
    :param D: the number of dimensions to optimize this function in
    :param n: the population size, number of particles
    :param alpha: learning parameter #1 (alpha is close to beta is close to 2)
    :param beta: learning parameter #2 (alpha is close to beta is close to 2)
    :param L: inertia weight for velocity
    :param gamma: regulates the number of particles to be substituted by
        crossover. Smaller values cause more particles to be substituted.
    :param max_iterations: the maximum number of iterations that the algorithm
        should run for (this is the stopping criterion)
    :param elite_count: the number of elite particles to keep in the population
    :param lower_bound: minimum value for xi
    :param upper_bound: maximum value for xi
    :return: x_best and f_best, the best solution found and its evaluation
    """

    # Initialization of positions X^0 using a random and uniformly
    # distributed Latin hypercube within the ranges of feasible values for each
    # input parameter.
    x = (lhs(D, samples=n, random_state=1234) * (upper_bounds - lower_bounds) + lower_bounds)

    # Initialization of velocities according to V^0 = (random_LHS(s, D) - X^0) / 2
    v = ((lhs(D, samples=n, random_state=1234) * (upper_bounds - lower_bounds) + lower_bounds) - x) / 2

    g_star = min(x, key=f)  # best particle at time t
    x_bests = {i: x[i] for i in range(n)}  # current best for particle i (i:x_best)
    xp_bests = {i: x[i] for i in range(n)}  # current best for particle i, but with parsimony (i:xp_best)

    for t in range(1, max_iterations + 1):
        for i in range(n):  # for each particle, do stuff
            e1 = np.random.uniform(size=D)
            e2 = np.random.uniform(size=D)

            # generate new velocity
            v[i] = L * v[i] + alpha * e1 * (g_star - x[i]) + beta * e2 * (x_bests[i] - x[i])

            # calculate new locations xi = xi + vi
            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], a_min=lower_bounds, a_max=upper_bounds)

            # update the current best for this particle
            if f(x[i]) < f(x_bests[i]):
                x_bests[i] = x[i]

            # TODO: update the parsimonious best for particle i


            # calculate pcrossover from equation (3)
            pcrossover = max(0.8 * math.exp(-gamma * t), 0.1)

            # TODO: crossover P_e to substitute P_w with new individuals


        # find the current global best g*
        g_star = min(x_bests.values(), key=f)

    # output the final results xi* and g*  (x_best, f_best?)
    return g_star, f(g_star)
