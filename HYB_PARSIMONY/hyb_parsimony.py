# ---------------------------------------------------------------------------------
# Hybrid Parsimony Algorithm Implementation
# ---------------------------------------------------------------------------------
#
# Description:
# This script implements the Hybrid Parsimony (HYB-PARSIMONY) algorithm, which combines
# the exploration mechanisms of Particle Swarm Optimization (PSO) with the exploitation
# capabilities of Genetic Algorithms (GA) to perform optimization tasks. The algorithm 
# is designed to optimize a given objective function by evolving a population of particles 
# that not only follow the best and personal best strategies from PSO but also undergo 
# crossover and mutation as in GAs to ensure diversity and avoid premature convergence.
#
# Features:
# - Hybrid Optimization: Utilizes PSO for global exploration with crossover and mutation
#   from GA for local exploitation.
# - Dynamic Adaptation: Includes mechanisms to dynamically adjust crossover rates and 
#   perform selective mutation to maintain diversity in the particle population.
# - Evaluator Integration: Uses a custom Evaluator class for assessing the fitness of 
#   particles, which is critical for guiding the optimization process.
#
# Key Components:
# - Particle: Represents a candidate solution in the search space with methods to update
#   its position based on velocity, personal best, and global best.
# - HybridParsimony: Orchestrates the optimization process by managing particles, performing
#   crossover and mutation, and selecting elite candidates for reproduction.
#
# Usage:
# - The script is intended to be used as part of a larger system where the objective function,
#   dimensionality of the problem, and specific parameters like population size and mutation 
#   rates are defined externally.
# - Execute the `solve()` method of the HybridParsimony class to start the optimization process.
#
# Dependencies:
# - numpy: Used for mathematical operations and managing array data.
# - pyDOE2: Provides tools for generating sampling plans using Latin Hypercube Sampling, which
#   is crucial for initializing particles in the solution space.
# - model_eval: A custom module that should define the Evaluator class used to assess the fitness
#   of solutions.
#
# Installation of Dependencies:
# - numpy: pip install numpy
# - pyDOE2: pip install pyDOE2
# - Ensure that the model_eval module is available in your environment as it must be imported.
#
# Example:
# An example usage scenario could involve optimizing hyperparameters for a machine learning model,
# where each particle represents a set of potential hyperparameters and the fitness is determined
# by the model's performance on validation data.
#
# ---------------------------------------------------------------------------------

import numpy as np
import math
from pyDOE2 import lhs
from model_eval import Evaluator

class HybridParsimony:
    """
    Implements the HYB-PARSIMONY algorithm. This algorithm is based on particle
    swarm optimization, but makes use of mutation and crossover from genetic
    algorithms.
    """
    def __init__(self, f, D, num_particles, max_iterations, alpha, beta, gamma, L, elite_count,
                 num_hyperparameters,evaluator:Evaluator, p_mutation=0.1, feat_mut_threshold=0.1,
                 not_muted=3) -> None:
        """
        Initializes an instance of the HYB-PARSIMONY algorithm using the given
        parameters.

        :param f: the objective function to optimize
        :param D: the number of dimensions to optimize this function in
        :param num_particles: the population size
        :param max_iterations: the number of iterations to run the algorithm for
        :param alpha: controls the influence of the global best value
        :param beta: controls the influence of each particle's personal best value
        :param gamma: regulates the number of particles to be substituted by
            crossover. Smaller values cause more particles to be replaced.
        :param L: the inertia weight for velocity updates
        :param elite_count: the number of elite particles to keep in the population
        :param num_hyperparameters: the number of hyperparameters in the
            solution vectors. The vectors are [[hyperparameters] + [features]].
        :param evaluator: the evaluator object
        """

        # Initialize the parameters
        self.f = lambda x: f(self.reverse_params(np.array(x)))
        self.D = D
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.num_hyperparameters = num_hyperparameters
        self.evaluator:Evaluator = evaluator
        # Tunable hyperparameters
        self.alpha = alpha  # Controls the influence of the global best value
        self.beta = beta  # Controls the influence of the personal best value
        self.L = L  # inertia weight
        self.gamma = gamma
        self.elite_count = elite_count

        self.p_mutation = p_mutation
        self.feat_mut_threshold = feat_mut_threshold
        self.not_muted = not_muted

        # Create the particles
        self.particles=[Particle(self.f, self.D, self.evaluator,
                                 self.alpha, self.beta,
                                 self.L,self.num_hyperparameters)
                        for i in range(self.num_particles)]

        # Initialize the current global best value among the particles
        self.global_best_val = self.particles[0].best_val
        self.global_best_f = self.particles[0].best_f

        # find the best value in the initial population
        for particle in self.particles:
            self.update_global_best(particle)

    def reverse_params(self, x):
        return np.hstack((x[:,self.num_hyperparameters:], x[:,:self.num_hyperparameters]))
    
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
        Selects the elite particles from the population based on their current
        function values. The number of elite particles is determined by the
        elite_count parameter of the class.

        This function performs an in-place sort of the current particles based
        on their current fitness values, in ascending order. This means that the
        particles with the best (lowest) fitness values are at the start of the
        list, and those with the worst values are at the end.

        :return: the elite particles
        """

        # Sort the particles by their function values in descending order
        # self.particles.sort(key=lambda p: p.current_f, reverse=True)

        # Sort the particles by their function values in ascending order
        self.particles.sort(key=lambda p: p.current_f, reverse=False)

        # Return the elite particles
        # return self.particles[self.elite_count:]  # for descending
        return self.particles[:self.elite_count]  # for ascending

        # return self.particles[:self.elite_count]


    def crossover_two(self, parent1, parent2):
        """
        Performs the crossover operation between two particles. Single-point
        crossover is carried out for both the features and the hyperparameters,
        using a random point to split the vectors.

        Particles are made up of [hyperparameters, features] in a single vector.

        :param parent1: the first parent particle
        :param parent2: the second parent particle
        :return: a new child particle
        """

        # retrieve hyperparameters and features
        parent1_hyperparameters = parent1.get_val()[:self.num_hyperparameters]
        parent1_features = parent1.get_val()[self.num_hyperparameters:]

        parent2_hyperparameters = parent2.get_val()[:self.num_hyperparameters]
        parent2_features = parent2.get_val()[self.num_hyperparameters:]

        # select crossover point
        features_len = len(parent1_features)
        features_crossover_point = np.random.randint(0, features_len)
        hyperparameters_len = len(parent1_hyperparameters)
        hyperparameters_crossover_point = np.random.randint(0, hyperparameters_len)

        # perform crossover for hyperparameters
        offspring1_hyperparameters = np.concatenate((parent1_hyperparameters[:hyperparameters_crossover_point],
                                                        parent2_hyperparameters[hyperparameters_crossover_point:]))
        offspring2_hyperparameters = np.concatenate((parent2_hyperparameters[:hyperparameters_crossover_point],
                                                        parent1_hyperparameters[hyperparameters_crossover_point:]))

        # now perform crossover for features
        offspring1_features = np.concatenate((parent1_features[:features_crossover_point],
                                              parent2_features[features_crossover_point:]))
        offspring2_features = np.concatenate((parent2_features[:features_crossover_point],
                                              parent1_features[features_crossover_point:]))

        # Randomly select which offspring to return
        # I don't want to return two particles, because what if we need to
        # replace an odd number of them? Then it wouldn't quite work out.
        if np.random.rand() < 0.5:
            chosen_offspring_val = np.concatenate((offspring1_hyperparameters, offspring1_features))
        else:
            chosen_offspring_val = np.concatenate((offspring2_hyperparameters, offspring2_features))

        # Build up the new particle
        new_particle = Particle(self.f, self.D, self.evaluator, self.alpha, self.beta,
                                self.L,self.num_hyperparameters, val=chosen_offspring_val)
        return new_particle


    def replace_bad_particles(self, t):
        """
        Performs the crossover step, using the elite particles to replace a
        worst-performing percentage of the population.

        :param t: the current iteration number (this matters for the crossover
            probability)
        :return: None
        """

        elitist_population = self.select_elite()
        p_crossover = max(0.8 * math.exp(-self.gamma * t), 0.1)

        # calculate how many particles to replace (using np.round)
        num_to_replace = int(np.round(p_crossover * self.num_particles))

        replacements = []
        for _ in range(num_to_replace):
            # randomly sample two particles from the elitist population
            # use them to generate a new offspring
            # add that offspring to replacements

            parent1, parent2 = np.random.choice(elitist_population, size=2, replace=False)

            # low chance to select a random particle for increased diversity
            if np.random.rand() < 0.1:
                parent2 = np.random.choice(self.particles)

            offspring = self.crossover_two(parent1, parent2)
            replacements.append(offspring)

        # At the end, replace the bottom p_crossover of the population with
        # their replacements.
        # Particles should already be sorted here from the call to select_elite.
        self.particles[-num_to_replace:] = replacements


    def mutate_particle(self, particle):
        """
        Mutates the hyperparameters and features (position) of a single
        particle. p_mutation represents the percentage of the hyperparameters to
        be mutated, feat_mut_threshold represents the probability of selecting a
        feature when mutating, and not_muted is the number of top best
        individuals that will not be mutated.

        The particle is mutated in place.

        :param particle: the particle to perform mutation on
        :return: the mutated particle
        """

        num_features = self.D - self.num_hyperparameters
        # break up the hyperparameters and features
        hyperparameters = particle.get_val()[:self.num_hyperparameters]
        features = particle.get_val()[self.num_hyperparameters:]

        # randomly select which hyperparameters to mutate
        num_hyp_to_mutate = int(np.round(self.p_mutation * self.num_hyperparameters))
        # make sure we're mutating at least one hyperparameter
        if num_hyp_to_mutate < 1:
            num_hyp_to_mutate = 1

        # if num_hyp_to_mutate < 1:  # don't mutate hyperparameters
        #     num_hyp_to_mutate = 0

        # choose the indices of the hyperparameters to mutate, then mutate them
        hyperparameters_indices = np.random.choice(self.num_hyperparameters, num_hyp_to_mutate, replace=False)
        for i in hyperparameters_indices:
            hyperparameters[i] = np.random.uniform(self.evaluator.template[i]['lower_bound'], self.evaluator.template[i]['upper_bound'])

        # randomly select which features to mutate
        num_fea_to_mutate = int(np.round(self.feat_mut_threshold * num_features))
        # make sure we're mutating at least one feature
        if num_fea_to_mutate < 1:
            num_fea_to_mutate = 1

        # if num_fea_to_mutate < 1:  # don't mutate features
        #     num_fea_to_mutate = 0

        # select indices as before, then mutate features
        features_indices = np.random.choice(num_features, num_fea_to_mutate, replace=False)
        for i in features_indices:
            features[i] = np.random.uniform(self.evaluator.template[i + self.num_hyperparameters]['lower_bound'],
                                                                      self.evaluator.template[i + self.num_hyperparameters]['upper_bound'])

        # update the particle and then force an update of its current and best values
        particle.val = np.concatenate((hyperparameters, features))


    def mutate_particles(self):
        """
        Performs an in-place mutation of all particles in the population, except
        for the top few particles.

        :return: None
        """

        # sort the particles by their current function values
        self.particles.sort(key=lambda p: p.current_f, reverse=False)

        # don't mutate the top few particles
        for particle in self.particles[self.not_muted:]:
            self.mutate_particle(particle)
            particle.clip()
        agents = [particle.reverse_params(particle.get_val()) for particle in self.particles]
        results=self.evaluator.execute(agents)
        for i,particle in enumerate(self.particles):
            particle.update_bests_and_current(results[i][0])


    def single_run(self):
        """
        Runs a single instance of the HYB-PARSIMONY algorithm to optimize the
        given objective function over the given dimensions using the given
        number of iterations.

        :return: the best position and f value found.
        """
        metrics=[]
        for t in range(1, self.max_iterations + 1):
            # PSO stuff: update each particle's position and velocity
            for particle in self.particles:
                particle.update(self.global_best_val)
            agents = [particle.reverse_params(particle.get_val()) for particle in self.particles]
            results=self.evaluator.execute(agents)
            for i,particle in enumerate(self.particles):
                particle.update_bests_and_current(results[i][0])
                self.update_global_best(particle)

            # Genetic algorithm stuff: crossover and mutation
            self.replace_bad_particles(t)  # crossover
            self.mutate_particles()  # mutation
            metrics.append([self.evaluator.best_agent_accuracy,sum([1 if val>0.5 else 0 for val in self.evaluator.best_agent])])
            
        for i in range(len(metrics)):
            print(f"Iteration {i + 1}:")
            print("Best agent accuracy:", metrics[i][0])
            print("Best agent num_features:", metrics[i][1])
            
        return self.global_best_val, self.global_best_f


    def solve(self):
        return self.single_run()



class Particle:
    """
    Represents a particle in the HYB-PARSIMONY algorithm. Each particle has a
    position, velocity, and personal best value. Additionally, each particle has
    an objective function to optimize, plus bounds for its position.

    For the purposes of the HYB-PARSIMONY algorithm, particles are made up of
    [hyperparameters, features] in a single vector.
    """
    def __init__(self, f, D, evaluator:Evaluator, alpha, beta, L,num_hyperparameters, val=None) -> None:
        """
        Initializes a single particle.

        :param f: the objective function to optimize
        :param D: the number of dimensions to optimize this function in
        :param evaluator: the evaluator object
        :param alpha: controls the influence of the global best value
        :param beta: controls the influence of each particle's personal best value
        :param L: the inertia weight for velocity updates
        """

        # Initialize the parameters
        self.f = f
        self.D = D
        self.evaluator=evaluator
        self.template=evaluator.template
        self.num_hyperparameters=num_hyperparameters
        # Tunable hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.L = L

        # self.val = np.random.uniform(lower_bounds, upper_bounds, size=D)
        # self.velocity = np.random.uniform(-1, 1, size=D)
        # Initialize the position and velocity from a latin hypercube sample
        lower_bounds = np.array([param['lower_bound'] for param in self.template])
        upper_bounds = np.array([param['upper_bound'] for param in self.template])
        if val is None:
            
            self.val = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()
        else:
            self.val = val

        self.velocity = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()
        
        self.best_val = self.val
        self.clip()
        self.best_f = f([self.val])[0][0]
        self.current_f = self.best_f

    def clip(self):
        clipped_val=self.evaluator.clip(self.reverse_params(self.val))
        num_features=self.D-self.num_hyperparameters
        self.val=np.hstack((clipped_val[num_features:],clipped_val[:num_features]))

    def reverse_params(self, x):
        return np.hstack((x[self.num_hyperparameters:],x[:self.num_hyperparameters]))

    def update_bests_and_current(self,function_val):
        """
        Forces an update of this particle's best position and best function
        value, as well as the current function value. This is useful when
        creating new particles, such as during crossover.

        :return: None
        """
        self.current_f = function_val  # keep track of the current function value, so we don't have to re-compute it later on
        if function_val < self.best_f:  # replace if better
            self.best_f = function_val
            self.best_val = self.val


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
        self.clip()
        # self.update_bests_and_current()


    def get_val(self):
        return self.val
