""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import scipy.io
import pandas as pd
import numpy as np
import copy


def data_from_file(file_name, **kwargs):

    if file_name.endswith('csv') or file_name.endswith('txt'):
        return process_csv(file_name, **kwargs)

    if file_name.endswith('mat'):
        return process_mat(file_name, **kwargs)

    raise NotImplementedError("i can't handle %s type files" % file_name)


def process_mat(file_name, **kwargs):

    print("loading %s... \n" % file_name)

    data_dict = scipy.io.loadmat("../../data/" + file_name)
    to_remove = []

    for key in data_dict:
        if "__" in key:
            to_remove.append(key)

    for key_to_rm in to_remove:
        del data_dict[key_to_rm]
        print("** deleted key: %s **" % key_to_rm)

    data = dict()

    if file_name == "fXSamples.mat":
        ix = kwargs.get('ix')
        data['ytrain'] = data_dict['x'][:, ix]
        return data

    elif file_name == "finPredProb.mat":
        my_array = data_dict['ttr'].flatten()
        data['ytrain'] = my_array
#         n_training = int(0.7 * len(my_array))
#         data['ytrain'] = my_array[:n_training]
#         data['ytest'] = my_array[n_training:]
        return data

    elif file_name == "mg.mat":
        data['ytrain'] = data_dict['t_tr'].flatten()
        data['ytest'] = data_dict['t_te'].flatten()
        return data

    elif file_name == "co2.mat":
        all_data = data_dict['co2'].flatten()
        data['ytrain'] = all_data[:500]
        data['ytest'] = all_data[500:]
        return data

    elif file_name == "sunspots.mat":
        n_training = data_dict['year'].flatten().tolist().index(1900)
        data['ytrain'] = data_dict['activity'].flatten()[:n_training].tolist()
        data['xtrain'] = np.arange(n_training).tolist()
#         data['xtest'] = data_dict['activity'].flatten()[n_training:]
#         data['ytest'] = np.arange(n_training, len(data_dict['year'].flatten()))
        data['xtest'] = []
        data['ytest'] = []
        to_attr = data_dict['activity'].flatten()[n_training:].tolist()
        for i in range(len(to_attr)):
            if i % 12 == 0:
                data['xtrain'].append(n_training + i)
                data['ytrain'].append(to_attr[i])
            else:
                data['xtest'].append(n_training + i)
                data['ytest'].append(to_attr[i])

        for key in data.keys():
            data[key] = np.array(data[key])

        return data

    else:
        raise NotImplementedError


def process_csv(file_name, **kwargs):

    my_dataframe = pd.read_csv("../../data/" + file_name)

    variable = kwargs.get('variable')

    if variable == 'tide':
        my_dataframe.rename(columns={'Tide height (m)': 'y',
                                     'True tide height (m)': 'ytruth',
                                     'Reading Date and Time (ISO)': 'x'},
                            inplace=True)
    elif variable == 'temperature':
        my_dataframe.rename(columns={'Air temperature (C)': 'y',
                                     'True air temperature (C)': 'ytruth',
                                     'Reading Date and Time (ISO)': 'x'},
                            inplace=True)
    else:
        raise ValueError(
            "Wrong predictor argument (%s), should be 'tide' or 'temperature'" % variable)

    my_dataframe['x'] = pd.to_datetime(my_dataframe['x'])
    t0 = copy.copy(my_dataframe['x'].ix[0])
    t0 = np.datetime64(t0)

    my_dataframe['x'] = my_dataframe['x'].apply(
        lambda x: (x - t0) / np.timedelta64(5, 'm'))

    print("done")
    print('-' * 50)

    print("creating training and testing dataframes...")

    testing_indices = my_dataframe['y'].index[
        my_dataframe['y'].apply(np.isnan)]

    n_rows = len(my_dataframe.index)
    training_indices = [i for i in range(n_rows) if i not in testing_indices]

    data_dict = dict()
    data_dict['ytrain'] = my_dataframe['y'][training_indices]
    data_dict['ytruthtrain'] = my_dataframe['ytruth'][training_indices]
    data_dict['xtrain'] = my_dataframe['x'][training_indices]
    data_dict['ytest'] = my_dataframe['y'][testing_indices]
    data_dict['ytruthtest'] = my_dataframe['ytruth'][testing_indices]
    data_dict['xtest'] = my_dataframe['x'][testing_indices]

    return data_dict
