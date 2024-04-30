# ===============================
# Tycho Bear
# CSCI.633.01
# Homework 2
# 2024-03-08
# ===============================

import math

"""
This file contains various test functions that are used in the homeworks.
"""

# Ackley's function
def ackley(x, D=2):
    """
    Python implementation of Ackley's function.

    For the homework, D = 2.

    x[i] should be within [-35, 35].

    The global minimum is located at the origin, i.e. x* = (0, ..., 0), with
    f(x*) = 0.

    :param x: the input vector.
    :param D: the number of elements in the Dx1 input vector.
    :return: the output of the function at the given x.
    """

    first_summation = 0
    for i in range(D):
        first_summation += x[i]**2

    radical_term = first_summation / D

    second_summation = 0
    for j in range(D):
        second_summation += math.cos(2 * math.pi * x[j])

    first_term = -20 * math.exp(-0.02 * math.sqrt(radical_term))
    second_term = math.exp(second_summation / D)

    return first_term - second_term + 20 + math.e


# Easom function
def easom(x, D=2):
    """
    Python implementation of the Easom function.

    For the homework, D = 2.

    x[i] should be within [-100, 100].

    The global minimum is located at x* = (π, π), with f(x*) = -1.

    :param x: the input vector.
    :param D: the number of elements in the Dx1 input vector.
    :return: the output of the function at the given x.
    """

    if len(x) != 2:
        print("\nError: input to the Easom function must have exactly two " +
              "values. (" + str(len(x)) + " found)")
        exit(-1)

    x1 = x[0]
    x2 = x[1]
    cos = -1 * math.cos(x1) * math.cos(x2)
    exp = math.exp(-1 * (x1 - math.pi)**2 - (x2 - math.pi)**2)

    return cos * exp


# Egg crate function
def egg_crate(x):
    """
    Python implementation of the two-dimensional egg crate function.

    For the homework, x1 and x2 should be within [-2π, 2π]. Otherwise, they
    should be within [-5, 5].

    The global minimum is located at x* = (0, 0) with f(x*) = 0.

    :param x: the input vector.
    :return: the output of the function at the given x.
    """

    if len(x) != 2:
        print("\nError: input to the egg crate function must have exactly two " +
              "values. (" + str(len(x)) + " found)")
        exit(-1)

    x1 = x[0]
    x2 = x[1]

    return x1**2 + x2**2 + 25*(math.sin(x1)**2 + math.sin(x2)**2)


# Four-peak function
def four_peak(x):
    """
    Python implementation of the two-dimensional four peak function.

    x1 and x2 should be within [-5, 5].

    The global maxima are located at x* = (0, 0) and x* = (0, -4) with f(x*) = 2.

    :param x: the input vector.
    :return: the output of the function at the given x.
    """

    if len(x) != 2:
        print("\nError: input to the four peak function must have exactly two " +
              "values. (" + str(len(x)) + " found)")
        exit(-1)

    x1 = x[0]
    x2 = x[1]

    first = math.e**(-(x1 - 4)**2 - (x2 - 4)**2)
    second = math.e**(-(x1 + 4)**2 - (x2 - 4)**2)
    third = 2 * (math.e**(-x1**2 - x2**2) + math.e**(-x1**2 - (x2 + 4)**2))

    return first + second + third


# Negative alpine function
def negative_alpine(x, D=2):
    """
    This is the cost function (the negative Alpine function) which the solver
    function will attempt to optimize.
    :param x: a list with two elements, i.e. [[0, 0]]
    :return: the z value of the [x, y] ([x1, x2]) coordinate vector
    """

    sum = 0
    for i in range(D):
        xi = x[0, i]
        term = abs((xi * math.sin(xi)) + 0.1*xi)
        sum += term

    sum *= -1
    return sum


# Rosenbrock's function
def rosenbrock(x, D=2):
    """
    Python implementation of Rosenbrock's function.

    For the homework, D = 2, and x[i] should be within [-2, 2]. Otherwise,
    x[i] should be within [-30, 30].

    The global minimum is located at x* = f(1, ..., 1), f(x*) = 0.

    :param x: the input vector.
    :param D: the number of elements in the Dx1 input vector.
    :return: the output of the function at the given x.
    """

    summation = 0
    for i in range(D - 1):
        # xi = x[i]
        term = ((x[i] - 1)**2) + (100 * (x[i+1] - x[i]**2)**2)
        summation += term

    return summation

