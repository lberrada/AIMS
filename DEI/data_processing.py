""" Data, Estimation & Inference module

data_processing.py : open and process data to proper format

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import pandas


def import_from_file(filename):
    """
    """
    
    my_dataframe = pandas.read_csv(filename)

    print(my_dataframe)
    
    return my_dataframe
    
    
