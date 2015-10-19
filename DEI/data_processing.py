""" Data, Estimation & Inference module

data_processing.py : open and process data to proper format

Author: Leonard Berrada
Date: 19 Oct 2015
"""

import pandas
import numpy as np


def process_from_file(filename):
    """
    """

    my_dataframe = pandas.read_csv(filename)

    print("data imported in dataframe")
    print('-' * 50)

    testing_indices = my_dataframe[
        'Tide height (m)'].index[my_dataframe['Tide height (m)'].apply(np.isnan)]

    n_rows = len(my_dataframe.index)
    training_indices = [i for i in range(n_rows) if i not in testing_indices]

    training_df = my_dataframe[
        ['Tide height (m)', 'Reading Date and Time (ISO)']].ix[training_indices]
    training_df.rename(
        columns={'Tide height (m)': 't', 'Reading Date and Time (ISO)': 'y'}, inplace=True)
    testing_df = my_dataframe[
        ['Tide height (m)', 'Reading Date and Time (ISO)']].ix[testing_indices]
    testing_df.rename(
        columns={'Tide height (m)': 't', 'Reading Date and Time (ISO)': 'y'}, inplace=True)

    print("training and testing dataframes created")
    print('-' * 50)
    
    print("Showing headers for verification:")
    print('\nTraining data :')
    print(training_df.head())
    print('\nTesting data :')
    print(testing_df.head())
    print('-' * 50)

    return training_df, testing_df
