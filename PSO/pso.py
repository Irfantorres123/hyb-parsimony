class ParticleSwarm:
    """
    Implements the particle swarm optimization algorithm
    """
    def __init__(self,f,max_iter,num_particles,lower_bound,upper_bound,alpha,beta) -> None:
        self.f=f
        self.max_iter=max_iter
        self.num_particles=num_particles
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.D=2
        self.alpha=alpha #Controls the influence of the global best value
        self.beta=beta #Controls the influence of the personal best value
        self.particles=[Particle(self.D,self.lower_bound,self.upper_bound,self.f,self.alpha,self.beta) for _ in range(self.num_particles)] # create the particles
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
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.global_best_val) # Update each particles position and velocity
                self.update_global_best(particle) # Update the global best value
        return self.global_best_val
    
    def solve(self):
        return self.single_run()
    


class Particle:
    """
    Represents a particle in the particle swarm optimization algorithm
    """
    def __init__(self,D,lower_bound,upper_bound,f,alpha,beta) -> None:
        self.D=D
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.val=np.random.uniform(lower_bound,upper_bound,size=(1,D))
        self.best_val=self.val
        self.best_f=f(self.val)
        self.f=f
        self.alpha=alpha
        self.beta=beta
        self.velocity=np.random.uniform(-1,1,size=(1,D))
    
    def update(self,global_best_val):
        """
        Update the particle's position and velocity
        """
        epsilon1=np.random.uniform(0,1) 
        epsilon2=np.random.uniform(0,1)
        self.velocity+=self.alpha*epsilon1*(global_best_val-self.val)+self.beta*epsilon2*(self.best_val-self.val) # update the velocity
        self.val=self.val+self.velocity # update the position
        self.val=np.clip(self.val,self.lower_bound,self.upper_bound) # clip the position to be within the bounds
        function_val=self.f(self.val) # evaluate the function at the new position
        if function_val<self.best_f: # if the function value at the new position is better than the best function value so far replace it
            self.best_f=function_val
            self.best_val=self.val
    
    def get_val(self):
        return self.val