""" Data, Estimation & Inference module

main.py : execution script

Author: Leonard Berrada
Date: 19 Oct 2015
"""

from data_processing import process_from_file

filename = 'sotonmet.txt'

training_df, testing_df = process_from_file(filename)





