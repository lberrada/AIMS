""" Description here

Author: Leonard Berrada
Date: 2 Nov 2015
"""

import sys

sys.path.append("../../DEI/src/")

from process_data import import_data

filename = "finPredProb.mat"
filename = "co2.mat"
filename = "mg.mat"

import_data(filename)