import numpy as np
from pyDOE2 import lhs
from benchmark_functions import easom

class HybridParsimony:
    """
    Implements the HYB-PARSIMONY algorithm. This algorithm is based on particle
    swarm optimization, but makes use of mutation and crossover from genetic
    algorithms.
    """
    def __init__(self, f, D, num_particles, max_iterations, lower_bounds, upper_bounds, alpha, beta, gamma, L, elite_count) -> None:
        """
        Initializes an instance of the HYB-PARSIMONY algorithm using the given
        parameters.

        :param f: the objective function to optimize
        :param D: the number of dimensions to optimize this function in
        :param num_particles: the population size
        :param max_iterations: the number of iterations to run the algorithm for
        :param lower_bounds: the minimum values for each dimension
        :param upper_bounds: the maximum values for each dimension
        :param alpha: controls the influence of the global best value
        :param beta: controls the influence of each particle's personal best value
        :param gamma: regulates the number of particles to be substituted by
            crossover. Smaller values cause more particles to be replaced.
        :param elite_count: the number of elite particles to keep in the population
        :param L: the inertia weight for velocity updates
        """

        # Initialize the parameters
        self.f = f
        self.D = D
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Tunable hyperparameters
        self.alpha = alpha  # Controls the influence of the global best value
        self.beta = beta  # Controls the influence of the personal best value
        self.L = L  # inertia weight
        self.gamma = gamma
        self.elite_count = elite_count

        # Create the particles
        self.particles=[Particle(self.f, self.D, self.lower_bounds,
                                 self.upper_bounds, self.alpha, self.beta,
                                 self.L)
                        for _ in range(self.num_particles)]

        # Initialize the current global best value among the particles
        self.global_best_val = self.particles[0].best_val
        self.global_best_f = self.particles[0].best_f

        # find the best value in the initial population
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
            self.global_best_f = particle.best_f
            self.global_best_val = particle.best_val


    def select_elite(self):
        """
        Selects the elite particles from the population.

        :return: the elite particles
        """


    def single_run(self):
        """
        Runs a single instance of the HYB-PARSIMONY algorithm to optimize the
        given objective function over the given dimensions using the given
        number of iterations.

        :return: The best position and f value found.
        """
        for t in range(1, self.max_iterations + 1):  # run for this number of iterations
            # Idea: update each particle's position and velocity, and then
            # update the global best
            for particle in self.particles:
                particle.update(self.global_best_val)
                self.update_global_best(particle)

            # Now select the elitist population for reproduction, and then
            # perform crossover and mutation
            # TODO: function call here

        return self.global_best_val, self.global_best_f


    def solve(self):
        return self.single_run()



class Particle:
    """
    Represents a particle in the HYB-PARSIMONY algorithm. Each particle has a
    position, velocity, and personal best value. Additionally, each particle has
    an objective function to optimize, plus bounds for its position.
    """
    def __init__(self, f, D, lower_bounds, upper_bounds, alpha, beta, L) -> None:
        """
        Initializes a single particle.

        :param f: the objective function to optimize
        :param D: the number of dimensions to optimize this function in
        :param lower_bounds: the minimum values for each dimension
        :param upper_bounds: the maximum values for each dimension
        :param alpha: controls the influence of the global best value
        :param beta: controls the influence of each particle's personal best value
        :param L: the inertia weight for velocity updates
        """

        # Initialize the parameters
        self.f = f
        self.D = D
        self.lower_bound = lower_bounds
        self.upper_bound = upper_bounds

        # Tunable hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.L = L

        # self.val = np.random.uniform(lower_bounds, upper_bounds, size=D)
        # self.velocity = np.random.uniform(-1, 1, size=D)
        # Initialize the position and velocity from a latin hypercube sample
        self.val = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()
        self.velocity = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()

        self.best_val = self.val
        self.best_f = f(self.val)
        self.current_f = self.best_f


    def update(self,global_best_val):
        """
        Updates this particle's velocity and position. This is done once per
        iteration (or generation) of the algorithm.

        :param global_best_val: the global best position found so far
        :return: None
        """

        epsilon1=np.random.uniform(0,1)
        epsilon2=np.random.uniform(0,1)

        # First update velocity, then update position using this new velocity
        # self.velocity+=self.alpha*epsilon1*(global_best_val-self.val)+self.beta*epsilon2*(self.best_val-self.val) # update the velocity
        self.velocity = self.L * self.velocity + self.alpha * epsilon1 * (
                    global_best_val - self.val) + \
                        self.beta * epsilon2 * (self.best_val - self.val)
        self.val = self.val + self.velocity
        # don't use this line of code, for some reason it produces bad results:
        # self.val += self.velocity

        # stay within bounds
        self.val=np.clip(self.val,self.lower_bound,self.upper_bound)

        # Then update this particle's best, as needed
        function_val = self.f(self.val)  # evaluate the function at the new position
        self.current_f = function_val  # keep track of the current function value, so we don't have to re-compute it later on
        if function_val < self.best_f:  # if the function value at the new position is better than the best function value so far replace it
            self.best_f = function_val
            self.best_val = self.val


    def get_val(self):
        return self.val
