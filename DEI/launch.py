""" Description here

Author: Leonard Berrada
Date: 23 Oct 2015
"""

from main import run
import csv

filename = 'sotonmet.txt'
sequential_mode = False

for estimator in ["MLE", "MAP"]:
    for variable in ["tide", "temperature"]:
        for op in ["+", "*"]:
            for use_kernels in ["exponential_quadratic" + op + "exponential_quadratic_2", 
                                "exponential_quadratic" + op + "periodic", 
                                "rational_quadratic" + op + "periodic",
                                "matern_12" + op + "periodic",
                                "matern_32" + op + "periodic"]:
                for use_means in ["constant", "constant + periodic"]:
                    try:
                        run(filename=filename,
                            variable=variable,
                            use_kernels=use_kernels,
                            estimator=estimator,
                            use_means=use_means,
                            sequential_mode=sequential_mode)
                    except:
                        print("damn")
                        with open("./out/results_v2.csv", 'a', newline='') as csvfile:
                            my_writer = csv.writer(csvfile, delimiter='\t',
                                                   quoting=csv.QUOTE_MINIMAL)
                            my_writer.writerow(["shit happened with following parameters:", 
                                                 variable,
                                                 use_kernels,
                                                 use_means,
                                                 estimator])
                    