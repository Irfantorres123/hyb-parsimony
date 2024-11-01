# ---------------------------------------------------------------------------------
# Particle Swarm Optimization (PSO) Implementation
# ---------------------------------------------------------------------------------
#
# Description:
# This script defines and implements the Particle Swarm Optimization algorithm for
# optimizing a given function over a specified domain. The algorithm simulates a swarm of
# particles moving through the problem space, adjusting their positions based on both
# personal and global best known positions.
#
# Components:
# - ParticleSwarm: A class representing the swarm of particles, which coordinates the
#   optimization process, updates global and individual bests, and handles the iteration
#   loop for convergence towards the optimum.
# - Particle: A class representing an individual particle in the swarm, which tracks its
#   own position, velocity, and best-known position. It adjusts its trajectory towards
#   its own best and the swarm's global best positions.
# - Evaluator: A separate module that likely includes functionality for evaluating
#   the fitness of particles, not defined in this script but expected to provide necessary
#   interfaces like evaluating accuracy and clipping values to maintain them within bounds.
#
# Usage:
# - Instantiate the ParticleSwarm class with a function to optimize, number of iterations,
#   number of particles, an evaluator instance, and specific coefficients that influence
#   the movement of particles.
# - Call the `solve` method on the ParticleSwarm instance to start the optimization process.
#
# Dependencies:
# - numpy: Utilized for numerical operations and array handling.
# - model_eval: Contains the Evaluator class with methods for evaluating and managing particle fitness.
# - pyDOE: Used for generating Latin Hypercube Samples which initializes particle positions within the space.
#
# Installation of dependencies:
# - numpy: pip install numpy
# - pyDOE: pip install pyDOE
# - model_eval: This appears to be a custom module, ensure it is available in your environment.
#
# ---------------------------------------------------------------------------------

import numpy as np
from model_eval import Evaluator
from pyDOE import lhs
class ParticleSwarm:
    """
    Implements the particle swarm optimization algorithm
    """
    def __init__(self,f,max_iter,num_particles,evaluator,alpha,beta,D) -> None:
        self.f=f
        self.max_iter=max_iter
        self.num_particles=num_particles
        self.evaluator:Evaluator=evaluator
        self.D=D
        self.alpha=alpha #Controls the influence of the global best value
        self.beta=beta #Controls the influence of the personal best value
        self.particles=[Particle(self.D,self.evaluator,f,self.alpha,self.beta) for _ in range(self.num_particles)] # create the particles
        self.global_best_val=self.particles[0].best_val
        self.global_best_f=self.particles[0].best_f
        for particle in self.particles:
            self.update_global_best(particle) # find the best value in the initial population
    
    def update_global_best(self,particle):
        """
        Update the global best value if the particle's best value is better
        """
        if particle.best_f<self.global_best_f:
            self.global_best_f=particle.best_f
            self.global_best_val=particle.best_val

    def single_run(self):
        """
        Run a single instance of the particle swarm optimization algorithm
        """
        metrics=[]
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.global_best_val) # Update each particles position and velocity
            agents=[particle.get_val() for particle in self.particles] # Get the positions of all the particles
            results=self.f(agents) # Evaluate the function at the positions of all the particles
            for i,particle in enumerate(self.particles):
                particle.update_best(results[i][0]) # Update the personal best value
                self.update_global_best(particle) # Update the global best value
            metrics.append([self.evaluator.best_agent_accuracy,sum([1 if val>0.5 else 0 for val in self.evaluator.best_agent])])

        for i in range(len(metrics)):
            print(f"Iteration {i + 1}:")
            print("Best agent accuracy:", metrics[i][0])
            print("Best agent num_features:", metrics[i][1])
            
        return self.global_best_val
    
    def solve(self):
        return self.single_run()
    


class Particle:
    """
    Represents a particle in the particle swarm optimization algorithm
    """
    def __init__(self,D,evaluator,f,alpha,beta) -> None:
        self.D=D
        self.evaluator=evaluator
        lower_bounds = np.array([param['lower_bound'] for param in self.evaluator.template])
        upper_bounds = np.array([param['upper_bound'] for param in self.evaluator.template])
        self.val = (lhs(D, samples=1) * (upper_bounds - lower_bounds) + lower_bounds).flatten()
        self.best_val=self.val
        self.best_f=f([self.val])[0][0]
        self.alpha=alpha
        self.beta=beta
        self.velocity=np.random.uniform(-1,1,size=(D))
    
    def update(self,global_best_val):
        """
        Update the particle's position and velocity
        """
        epsilon1=np.random.uniform(0,1) 
        epsilon2=np.random.uniform(0,1)
        self.velocity+=self.alpha*epsilon1*(global_best_val-self.val)+self.beta*epsilon2*(self.best_val-self.val) # update the velocity
        self.val=self.val+self.velocity # update the position
        
        self.clip() # clip the position to the bounds

    def update_best(self,function_val):
        if function_val<self.best_f: # if the function value at the new position is better than the best function value so far replace it
            self.best_f=function_val
            self.best_val=self.val

    def clip(self):
        """
        Clip the particle's position to the bounds
        """

        self.val=self.evaluator.clip(self.val)

    def get_val(self):
        return self.val