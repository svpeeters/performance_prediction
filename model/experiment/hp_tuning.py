import os
import sys

import numpy
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

from model.helper import HPLogger, NumpyEncoder

sys.path.append("..")

import json


import pandas
from sklearn.model_selection import KFold
from skopt import gp_minimize, dump
from skopt.space import Integer, Real

from skopt.utils import use_named_args

from model.dnn import DNNRegressor, MonotonicBatchDNNRegressor, IntervalRegressorMAE, \
    IntervalRegressorMAEMonotonic

import time

from model.transformer import IntervalTargetTransformer, MeanTransformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SCENARIO_1 = True
SCENARIO_2 = True

TUNE_INTERVAL = True
TUNE_BASELINE_MEAN = True

# monotonic additions

TUNE_BASELINE_MEAN_MONO = True
TUNE_INTERVAL_MONO = True

CV = 5
REPETITIONS = 2
N_JOBS = 10
SAMPLE_SIZE = 7500
ITERATIONS_PER_SPACE = 125
LOG_PATH = 'hp_tuning/%s/'
RES_PATH = 'hp_tuning/tuning_results.json'

dnn_param_names = ['num_dense_layers', 'num_input_nodes', 'num_dense_nodes', 'ratio_dropout', 'penalty_weight']


def prepare_params(**params):
    reg_params = {}
    trans_params = {}

    for param in params:
        if param in dnn_param_names:
            reg_params[param] = params[param]
        else:
            trans_params[param] = params[param]
    return {'reg': reg_params,
            'trans': trans_params}


def score_fold(X_train, y_train, X_test, y_test, model_creator, params):
    # create model
    params['batch_size'] = 64
    params['max_epochs'] = 1000
    model = model_creator()
    model.set_params(**params)

    # train model with params
    model.fit(X_train, y_train)

    # evaluate model on tests set with mean absolute error
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    return mae


def scorer(params, X_trans, y_trans, X, y, y_mean, model_creater, transform_creater):
    tf.keras.backend.clear_session()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # apply transform if needed
    transform = transform_creater()
    if transform is not None:
        transform.set_params(**params['trans'])
        X_trans, y_trans = transform.transform(X, y)
        X_trans = pandas.DataFrame(X_trans)
        y_trans = pandas.DataFrame(y_trans)

    process_params = []
    for train, test in kf.split(X):
        X_train = pandas.DataFrame(X_trans.iloc[train]).reset_index(drop=True)
        y_train = pandas.DataFrame(y_trans.iloc[train]).reset_index(drop=True)

        # validation sets for early stopping of training
        X_test = pandas.DataFrame(X.iloc[test]).reset_index(drop=True)
        y_test = pandas.DataFrame(y_mean.iloc[test]).reset_index(drop=True)

        for _ in range(REPETITIONS):
            process_params.append([
                X_train, y_train, X_test, y_test, model_creater, params['reg']
            ])

    split_results = Parallel(n_jobs=N_JOBS)(delayed(score_fold)(*params) for params in process_params)
    results_array = numpy.array(split_results).reshape(-1, REPETITIONS)
    return numpy.mean(results_array, axis=1)



def tune_model(X_trans: pandas.DataFrame, y_trans: pandas.DataFrame, X, y,  model_creater, transform_creater, search_space, log_path: str):
    # create log folder if it not exists
    dir_path = os.path.dirname(log_path + 'mock.file')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log_columns = [elem.name for elem in search_space] + ['time', 'std', 'split_scores' , 'score']
    log_monitor_path = log_path + "log.csv"
    monitor = HPLogger(log_monitor_path, log_columns)

    param_list = []

    # mean y for the test set
    y_mean = y.mean(axis=1)

    @use_named_args(search_space)
    def evaluate(**params):
        # prepare parms
        p = prepare_params(**params)
        # apply params to transform and model
        start = time.time()

        result_splits = scorer(p, X_trans, y_trans, X, y, y_mean, model_creater, transform_creater)

        score = numpy.mean(result_splits)
        variance = numpy.std(result_splits)

        log_row = {
            **p['reg'],
            **p['trans'],
            'time': time.time() - start,
            'split_scores': result_splits,
            'std': variance,
            'score': score
        }
        monitor.write_row(log_row)

        param_list.append((p, score))
        return score

    total_iterations = ITERATIONS_PER_SPACE * len(search_space)
    initial_points = int(total_iterations * 0.15)
    print(total_iterations)
    print(initial_points)
    result = gp_minimize(evaluate, search_space,
                         n_calls=total_iterations,
                         n_initial_points=initial_points,
                         n_jobs=-1,
                         verbose=True,
                         random_state=27,
                         model_queue_size=10)
    del result.specs['args']['func']
    dump(result, filename=log_path + 'result.opt', store_objective=False)
    return param_list


def tune(dataset_path, log_path, res_path):
    X = pandas.read_csv(dataset_path % "X_train.csv")
    y = pandas.read_csv(dataset_path % "y_train.csv")

    X.fillna(0, inplace=True)

    # suffle X and y since BayesSearchCV does not shuffle automatically
    X = X.sample(n=SAMPLE_SIZE, random_state=27021996)
    y = y.reindex(X.index)

    # reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    num_datasets = 1#X['dataset'].nunique()

    print("num_datasets: %d" % num_datasets)

    search_space_nn = [
        Integer(low=1, high=3, name='num_dense_layers'),
        Integer(low=10, high=350, name='num_input_nodes'),
        Integer(low=5, high=200, name='num_dense_nodes'),
        Real(low=0.0, high=0.5, name='ratio_dropout')
    ]


    monotonic_increasing = ['cpu_Frequency', 'cpu_Turbo Clock', 'cpu_Multiplier',
                            'gpu_Base Clock', 'gpu_Boost Clock', 'gpu_Bandwidth']
    monotonic_decreasing = ['resolution', 'setting']

    search_space_mon_penalty = [Real(low=0.0, high=100.0, name='penalty_weight')]

    tuning_result_dict = {}

    if TUNE_INTERVAL_MONO:
        # monotonic increasing features
        monotonic_increasing_indices = [X.columns.get_loc(name) for name in monotonic_increasing]
        monotonic_decreasing_indices = [X.columns.get_loc(name) for name in monotonic_decreasing]
        # monotonic decreasing features

        def model_creater():
            MonotonicBatchDNNRegressor.mon_increasing = monotonic_increasing_indices
            MonotonicBatchDNNRegressor.mon_decreasing = monotonic_decreasing_indices
            return IntervalRegressorMAEMonotonic()

         # pre transform since the intervals are independent of the params
        trans = IntervalTargetTransformer()
        X_trans, y_trans = trans.transform(X, y)
        X_trans = pandas.DataFrame(X_trans)
        y_trans = pandas.DataFrame(y_trans)
        def transform_creater():
            return None

        res_list = tune_model(X_trans, y_trans, X, y, model_creater, transform_creater, search_space_nn + search_space_mon_penalty,
                              log_path % 'interval_monotonic')
        tuning_result_dict['interval_monotonic'] = sorted(res_list, key=lambda tuple: tuple[1])[0][0]

    if TUNE_BASELINE_MEAN_MONO:
        # monotonic increasing features
        monotonic_increasing_indices = [X.columns.get_loc(name) for name in monotonic_increasing]
        monotonic_decreasing_indices = [X.columns.get_loc(name) for name in monotonic_decreasing]
        # monotonic decreasing features

        MonotonicBatchDNNRegressor.mon_increasing = monotonic_increasing_indices
        MonotonicBatchDNNRegressor.mon_decreasing = monotonic_decreasing_indices

        def model_creater():
            MonotonicBatchDNNRegressor.mon_increasing = monotonic_increasing_indices
            MonotonicBatchDNNRegressor.mon_decreasing = monotonic_decreasing_indices
            return MonotonicBatchDNNRegressor()

        trans = MeanTransformer()
        X_trans, y_trans = trans.transform(X, y)
        X_trans = pandas.DataFrame(X_trans)
        y_trans = pandas.DataFrame(y_trans)

        def transform_creater():
            return None

        res_list = tune_model(X_trans, y_trans, X, y, model_creater, transform_creater, search_space_nn + search_space_mon_penalty,
                              log_path % 'mean_monotonic')
        tuning_result_dict['mean_monotonic'] = sorted(res_list, key=lambda tuple: tuple[1])[0][0]

    # interval
    if TUNE_INTERVAL:

        def model_creater():
            return IntervalRegressorMAE()

         # pre transform since the intervals are independent of the params
        trans = IntervalTargetTransformer()
        X_trans, y_trans = trans.transform(X, y)
        X_trans = pandas.DataFrame(X_trans)
        y_trans = pandas.DataFrame(y_trans)

        def transform_creater():
            return None

        res_list = tune_model(X_trans, y_trans, X, y, model_creater, transform_creater, search_space_nn, log_path % 'interval')
        tuning_result_dict['interval'] = sorted(res_list, key=lambda tuple: tuple[1])[0][0]

    # baseline mean
    if TUNE_BASELINE_MEAN:

        def model_creater():
            return DNNRegressor()

        trans = MeanTransformer()
        X_trans, y_trans = trans.transform(X, y)
        X_trans = pandas.DataFrame(X_trans)
        y_trans = pandas.DataFrame(y_trans)

        def transform_creater():
            return None

        res_list = tune_model(X_trans, y_trans, X,y, model_creater, transform_creater, search_space_nn, log_path % 'mean')
        tuning_result_dict['mean'] = sorted(res_list, key=lambda tuple: tuple[1])[0][0]


    # dump result dict to json
    with open(res_path, 'w') as result_file:
        result_file.write(json.dumps(tuning_result_dict, cls=NumpyEncoder, indent=4))


if __name__ == '__main__':
    # tune scenario 1
    if SCENARIO_1:
        tune(dataset_path='../data/case_study/scenario_1/%s', log_path='hp_tuning/scenario_1/%s/',
             res_path='hp_tuning/scenario_1.json')
    # tune scenario 2
    if SCENARIO_2:
        tune(dataset_path='../data/case_study/scenario_2/%s', log_path='hp_tuning/scenario_2/%s/',
             res_path='hp_tuning_cs/scenario_2.json')
