""" Description here

Author: Leonard Berrada
Date: 20 Oct 2015
"""

from main import run

# MLE Tide
params = [0.03, 0.1, 5e02, 9.4, 3e2, 2.9, 1, 1.5e2]
run('sotonmet.txt', 'tide', 'matern_32 * periodic', 'constant+periodic', 'MLE', False, params)

# MAP Tide
# params = [0.025, 0.71, 1.8e3 , 0.71, 3e2, 3]
# run('sotonmet.txt', 'tide', 'matern_12 * periodic', 'constant', 'MAP', False, params)

# MLE temp
# params = [0.25, 0.47, 3.8, 2.9, 41, 12]
# run('sotonmet.txt', 'temperature', 'exponential_quadratic + exponential_quadratic_2', 'constant', 'MLE', False, params)

# MAP temp
# params = [0.16, 1.5, 2.2e2, 1.9, 5.9e02, 12]
# run('sotonmet.txt', 'temperature', 'matern_12 * periodic', 'constant', 'MAP', False, params)
