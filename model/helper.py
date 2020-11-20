import csv
import json
from typing import List

import numpy
import numpy as np
import pandas
import tensorflow as tf


def modified_z_score(sample: numpy.ndarray):
    """
    Computes the modified z_score for each element of the sample in the numpy array sample.
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    @param sample: numpy array for which the modified z-score will be computed
    @return: numpy array with modified z-score for each element of sample
    """
    median = numpy.median(sample)
    mad = numpy.median(numpy.absolute(sample - median))
    numerator = 0.6745 * (sample - median)
    score = numpy.zeros_like(sample, dtype=numpy.float32)
    for index in range(len(sample)):
        if mad == 0 and numerator[index] == 0:
            score[index] = 0
        elif mad == 0 and numerator[index] != 0:
            score[index] = numpy.inf
        else:
            score[index] = numerator[index] / mad
    return score


def monotoncity_metric(model, k: int, k_resolution: numpy.array, X: numpy.array, mon_increasing: bool,
                       k_space: numpy.array = None):
    """
    Implementation of monotonicity metrix M_k.
    @param model: (tensorflow) model to evaluate monotonicity metric for
    @param k: index of feature k
    @param k_resolution: how many points will be used to check monotonicity
    @param X: 2 dimensional numpy array contaning input instances
    @param mon_increasing: true = k-th feature is monotone increasing, false = k-th feature is monotone decreasing
    @param k_space: space containing all values for the k-th feature (if this is None k-resolution will be used to
    create the space to check monotonicity behaviour)
    @return: monotonicity metric for the k-th feature
    """

    # convert to numpy if input is a dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.to_numpy()

    # if k_space is none generate based on min and max
    if k_space is None:
        X_k_column = X[:, k]
        k_space = numpy.linspace(numpy.min(X_k_column), numpy.max(X_k_column), k_resolution)

    # iterate over each point in data
    n = len(X)
    # create for each point in X each variation of k for
    delta = numpy.repeat(X, axis=0, repeats=len(k_space))
    k_space_column = numpy.concatenate([k_space] * n).flatten()
    delta[:, k] = k_space_column

    # use gradient to assess the monotonicity of the k'th feature
    pred = model(delta)
    if (isinstance(pred, tf.Tensor)):
        pred = pred.numpy()
    # reshape pred such that each pred for one data point is in one row
    pred_reshape = pred.reshape((n, len(k_space)))

    # check if the k-th feature is monotonic for the space and on input instance
    diff = numpy.diff(pred_reshape, axis=1)
    if mon_increasing:
        is_mon = numpy.all(diff >= 0.0, axis=1)
    else:
        is_mon = numpy.all(diff <= 0.0, axis=1)

    return numpy.count_nonzero(is_mon) / n


class HPLogger():
    """
        Logger for hyperparameter tuning
    """
    def __init__(self, log_path: str, fieldnames: List):
        self._log_path = log_path
        self._fieldnames = fieldnames
        with open(self._log_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames)
            writer.writeheader()

    def write_row(self, log_dict: dict):
        with open(self._log_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, self._fieldnames)
            writer.writerow(log_dict)


class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder for numpy data types to save them as a json
    """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)