# -*- coding: utf-8 -*-
"""
GPR
===========

This example illustrates the use of GPs

"""

# Author: Chester Holtz
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.gp import TimeSeriesGPR

numpy.random.seed(0)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)

clf = TimeSeriesGPR()
clf.fit(X_train, y_train)
print("Correct regression rate:", clf.score(X_test, y_test))