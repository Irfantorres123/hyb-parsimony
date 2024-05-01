import numpy as np

def particle_swarm_optimization(f, D, n, alpha, beta, max_iterations, lower_bound, upper_bound):
    """
    Implements the particle swarm optimization algorithm from Figure 7.2's pseudocode.

    :param f: the objective function to optimize
    :param D: the number of dimensions to optimize this function in
    :param n: the population size
    :param alpha: learning parameter #1 (alpha is close to beta is close to 2)
    :param beta: learning parameter #2 (alpha is close to beta is close to 2)
    :param max_iterations: the maximum number of iterations that the algorithm
        should run for (this is the stopping criterion)
    :param lower_bound: minimum value for xi
    :param upper_bound: maximum value for xi
    :return: x_best and f_best, the best solution found and its evaluation
    """

    # initialize locations xi and vi of n particles
    x = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, D))
    v = np.zeros(shape=(n, D))
    g_star = min(x, key=f)  # best particle at time t
    x_bests = {i: x[i] for i in range(n)}  # current best for particle i (i:x_best)
    t = 0  # time, # of iterations

    while t < max_iterations:
        # for i in range(n):
        # for loop over all n particles and all d dimensions
        for i in range(n):
            e1 = np.random.uniform(size=D)
            e2 = np.random.uniform(size=D)
            # L = 0.85  # inertia weight
            L = 0.7  # inertia weight
            # L = 0.5  # inertia weight
            # generate new velocity vi using equation (7.1)
            v[i] = L * v[i] + alpha * e1 * (g_star - x[i]) + beta * e2 * (x_bests[i] - x[i])

            # calculate new locations xi = xi + vi
            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], a_min=lower_bound, a_max=upper_bound)

            # evaluate objective function at new location for x[i] and update
            # the current best for this particle
            if f(x[i]) < f(x_bests[i]):
                x_bests[i] = x[i]

        # find the current global best g*
        g_star = min(x_bests.values(), key=f)
        t += 1

    # output the final results xi* and g*  (x_best, f_best?)
    return g_star, f(g_star)
