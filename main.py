# ---------------------------------------------------------------------------------
# Optimization Algorithm Testing Script
# ---------------------------------------------------------------------------------
#
# Description:
# This script is dedicated to the testing of three distinct optimization algorithms:
# Particle Swarm Optimization (PSO), Genetic Algorithm, and a Hybrid Algorithm that 
# combines features of multiple optimization strategies. It utilizes the testing 
# functions imported from the 'testing' package to evaluate each algorithm's performance.
#
# Structure:
# - The script starts by testing the PSO algorithm, followed by the Genetic Algorithm,
#   and concludes with the Hybrid Algorithm.
# - Each algorithm is tested using a dedicated function that is imported from respective 
#   modules within the 'testing' package.
# - Results for each test are printed sequentially, with clear separation and labeling
#   to ensure readability and ease of analysis.
#
# Usage:
# Run this script to execute the tests for each algorithm. Review the console output
# to analyze the performance characteristics and outcomes of each test.
#
# Note:
# Ensure that the 'testing' package and its submodules are correctly installed and
# accessible in your Python environment before running this script.
#
# ---------------------------------------------------------------------------------

from testing.geneticAlgotesting import test as genetic_test
from testing.HybridAlgotesting import test as hybrid_test
from testing.PSOAlgotesting import test as pso_test
print("Testing PSO")
pso_test()
print("\n\n\n")
print("Testing Genetic Algorithm")
genetic_test()
print("\n\n\n")
print("Testing Hybrid Algorithm")
hybrid_test()
