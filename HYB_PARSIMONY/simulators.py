# ===============================
# Tycho Bear
# CSCI.633.01
# Homework 4
# 2024-04-24
# ===============================

"""
This file contains the simulation for question 3, called in q3a and q3b.
"""

import numpy as np


def two_hyperparameter_grid_search(n_values, values_1, values_2, num_trials,
                                   solve, f, D, max_iterations, lower_bound,
                                   upper_bound):
    """
    Runs the simulation.

    :param num_trials: the number of trials to perform.
    :return: a 2-dimensional array containing the experimental data. This will
    be turned into copy-and-pastable LaTeX.
    """

    data = []

    for n in n_values:  # for each population size
        table_data_body = []
        print()
        for value1 in values_1:
            for value2 in values_2:  # for grid search
                found_optima = []  # for the f(x) from each trial
                for _ in range(num_trials):  # use these parameters for 30 trials

                    # For each trial, we find a solution and add it to the list
                    (x_best, f_best) = solve(f, D, n, value1, value2, max_iterations,
                                           lower_bound, upper_bound)
                    found_optima.append(f_best)

                # Calculate the mean and standard deviation of the values for this
                # parameter combination after 30 trials
                these_parameters_mean = np.mean(found_optima)
                these_parameters_std = np.std(found_optima)

                print("mean for n =", n, "alpha =", value1, ", gamma =",
                      value2, ":", "Mean optima:", f'{these_parameters_mean:.20f}')

                table_data_body.append(these_parameters_mean)
                table_data_body.append(these_parameters_std)

        # population size + data for different parameter combinations using it
        data.append([n] + table_data_body)

    return data


def two_hyperparameter_simulation(n_values, value_pairs, num_trials, solve, f, D,
                                  max_iterations, lower_bound, upper_bound):
    """
    Runs the simulation.

    :param num_trials: the number of trials to perform.
    :return: a 2-dimensional array containing the experimental data. This will
    be turned into copy-and-pastable LaTeX.
    """

    data = []

    for n in n_values:  # for each population size
        table_data_body = []
        print()
        for value1, value2 in value_pairs:  # for 3 combinations of (F, Cr)
            found_optima = []  # for the f(x) from each trial
            for _ in range(num_trials):  # use these parameters for 30 trials

                # For each trial, we find a solution and add it to the list
                (x_best, f_best) = solve(f, D, n, value1, value2, max_iterations,
                                       lower_bound, upper_bound)
                found_optima.append(f_best)

            # Calculate the mean and standard deviation of the values for this
            # parameter combination after 30 trials
            these_parameters_mean = np.mean(found_optima)
            these_parameters_std = np.std(found_optima)

            print("mean for n =", n, "alpha =", value1, ", gamma =",
                  value2, ":", "Mean optima:", f'{these_parameters_mean:.20f}')

            table_data_body.append(these_parameters_mean)
            table_data_body.append(these_parameters_std)

        # population size + data for different parameter combinations using it
        data.append([n] + table_data_body)

    return data


def three_hyperparameter_grid_search(n_values, values_1, values_2, values_3,
                                     num_trials, solve, f, D, max_iterations,
                                     lower_bound, upper_bound):
    """
    Runs the simulation.

    :param num_trials: the number of trials to perform.
    :return: a 2-dimensional array containing the experimental data. This will
    be turned into copy-and-pastable LaTeX.
    """

    data = []

    for n in n_values:  # for each population size
        table_data_body = []
        print()
        for value1 in values_1:
            for value2 in values_2:
                for value3 in values_3:
                    found_optima = []  # for the f(x) from each trial
                    for _ in range(
                            num_trials):  # use these parameters for 30 trials

                        # For each trial, we find a solution and add it to the list
                        (x_best, f_best) = solve(f, D, n, value1, value2, value3,
                                                 max_iterations, lower_bound, upper_bound)
                        found_optima.append(f_best)

                    # Calculate the mean and standard deviation of the values for this
                    # parameter combination after 30 trials
                    these_parameters_mean = np.mean(found_optima)
                    these_parameters_std = np.std(found_optima)

                    print("mean for n =", n, "alpha =", value1, ", beta =",
                          value2, "gamma =", value3, ":", "Mean optima:",
                          f'{these_parameters_mean:.20f}')

                    table_data_body.append(these_parameters_mean)
                    table_data_body.append(these_parameters_std)

        # population size + data for different parameter combinations using it
        data.append([n] + table_data_body)

    return data


def three_hyperparameter_simulation(n_values, value_triplets, num_trials, solve, f, D,
                                  max_iterations, lower_bound, upper_bound):
    """
    Runs the simulation.

    :param num_trials: the number of trials to perform.
    :return: a 2-dimensional array containing the experimental data. This will
    be turned into copy-and-pastable LaTeX.
    """

    data = []

    for n in n_values:  # for each population size
        table_data_body = []
        print()
        for value1, value2, value3 in value_triplets:  # run through each triplet of values
            found_optima = []  # for the f(x) from each trial
            for _ in range(num_trials):  # use these parameters for 30 trials

                # For each trial, we find a solution and add it to the list
                (x_best, f_best) = solve(f, D, n, value1, value2, value3,
                                         max_iterations, lower_bound, upper_bound)
                found_optima.append(f_best)

            # Calculate the mean and standard deviation of the values for this
            # parameter combination after 30 trials
            these_parameters_mean = np.mean(found_optima)
            these_parameters_std = np.std(found_optima)

            print("mean for n =", n, "| alpha =", value1, ", beta =", value2,
                  "gamma =", value3, ":", "Mean optima:", f'{these_parameters_mean:.20f}')

            table_data_body.append(these_parameters_mean)
            table_data_body.append(these_parameters_std)

        # population size + data for different parameter combinations using it
        data.append([n] + table_data_body)

    return data
