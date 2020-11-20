from typing import Tuple

import numpy
import pandas

from model.helper import modified_z_score


class TargetTransformer():
    """
    Definition of methods that a TargetTransformer has to implement. The Transformer MeanTransformer,
    IntervalTargetTransformer, and HistogramTargetTransformer inherit from this class and override the
    methods.
    """
    def transform(self, X, y) -> Tuple:
        """
        Transform the target variable of the observations y. X should not be altered since a TargetTransformer should
        only alter the target.
        @param X: Dataframe containing the input instances
        @param y: Dataframe containing the target instances (multisets)
        @return:
        """
        return None, None

    def get_params(self, deep=True):
        """
        Returns the params of the TargetTransformer.
        @param deep: not used
        @return:
        """
        # no parameters
        return {}

    def set_params(self, deep=True, **params):
        """
        Sets the paramters of the TargetTransformer. Paramters depend on the logic that the transformer is implementing.
        @param deep:
        @param params:
        @return:
        """
        # no parameters
        return self


class MeanTransformer(TargetTransformer):
    """
    A transformer that transform a multiset using the arithmetic mean
    """

    def transform(self, X: pandas.DataFrame, y: pandas.DataFrame) -> Tuple[numpy.array, numpy.array]:
        # apply mean to each row of the y df
        y_mean = y.mean(axis=1, skipna=True)
        return X.to_numpy(dtype=numpy.float32), y_mean.to_numpy(dtype=numpy.float32).reshape(-1, 1)


class IntervalTargetTransformer(TargetTransformer):
    """
    A transformer that uses the interval approach to transform a multiset. In short: modified Z-scrore is used
    to remove outliers from the multiset. Afterwards min and max are taken as the interval boundaries to obtain an
    interval from a multiset.
    """

    def transform(self, X, y) -> Tuple:
        """
        Transform multisets in y to intervals
        @param X: input instances
        @param y: target instances (multisets)
        @return: numpy array of X (unaltered) interval boundaries as numpy array with shape (# of instances,2)
        """
        y_interval = numpy.zeros((len(y), 2), dtype=numpy.float32)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for index, row in y.iterrows():
            if row.count() == 1:
                val = row.iloc[0]
                y_interval[index][0] = val
                y_interval[index][1] = val
            else:
                sample = row.dropna()

                abs_mod_z_score = numpy.absolute(modified_z_score(sample))
                final_sample = sample[abs_mod_z_score <= 3.5]

                y_interval[index][0] = min(final_sample)
                y_interval[index][1] = max(final_sample)
        return X.to_numpy(dtype=numpy.float32), y_interval