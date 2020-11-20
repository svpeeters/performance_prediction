import json
import os
import sys

from model.helper import monotoncity_metric


sys.path.append("..")

from model.dnn import DNNRegressor, MonotonicBatchDNNRegressor, IntervalRegressorMAE, \
    IntervalRegressorMAEMonotonic

import pandas
import numpy

import tqdm

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from model.transformer import IntervalTargetTransformer, MeanTransformer

import statistics

from joblib import Parallel, delayed

CV_5 = True
N_JOBS = 10

SAMPLE_SIZE = 7500
SAMPLE_SIZE_TEST = 7500

SCENARIO_1 = True
SCENARIO_2 = True

MONOTONOCITY_METRIC = True

EVAL_INTERVAL = True
EVAL_BASELINE_MEAN = True

EVAL_INTERVAL_MONO = True
EVAL_BASELINE_MEAN_MONO = True

INNER_REPS = 4

repetition_seeds = [
    428145165,
    610683743,
    228104079,
    321908317,
    838668603,
    781147312,
    337250001,
    565410184,
    549457808,
    700385509
]

monotonic_increasing = ['cpu_Frequency', 'cpu_Turbo Clock', 'cpu_Multiplier',
                        'gpu_Base Clock', 'gpu_Boost Clock', 'gpu_Bandwidth']
monotonic_decreasing = ['resolution', 'setting']

monotonic = monotonic_increasing + monotonic_decreasing

monotonic_metric_space = {'setting': numpy.array([0, 1, 2, 3])}
monotonic_indices_dict = {}
monotonic_increasing_indices = []
monotonic_decreasing_indices = []


def eval_fold(X_train, y_train, X_val, y_val, X_test, y_test, model_creator, split_log_path, model_log_path):
    # create model
    model = model_creator()

    train_args = {
        'X_val': X_val,
        'y_val': y_val,
        'epochs': 10000,
        'batch_size': 64
    }

    hist = model.fit(X_train, y_train, **train_args)
    hist_df = pandas.DataFrame(hist.history)
    hist_df.to_csv(split_log_path + ".csv", index=False)

    # evaluate model on tests set with mean absolute error
    y_pred = model.predict(X_test)

    y_test_np = y_test.to_numpy(dtype=numpy.float32).flatten()
    mae = mean_absolute_error(y_test_np, y_pred)

    mks: dict = {}
    if MONOTONOCITY_METRIC:
        for feature in monotonic:
            idx = monotonic_indices_dict[feature]
            increasing = feature in monotonic_increasing

            mk_tests = []
            mk_trains = []

            test_chunks = numpy.array_split(X_test, max(1, len(X_test) // 2500))
            train_chunks = numpy.array_split(X_train, max(1, len(X_train) // 2500))

            for chunk in train_chunks:
                if feature in monotonic_metric_space:
                    space = monotonic_metric_space[feature]
                    mk_train = monotoncity_metric(model.model, idx, 0, chunk, mon_increasing=increasing, k_space=space)
                else:
                    mk_train = monotoncity_metric(model.model, idx, 200, chunk, mon_increasing=increasing)
                mk_trains.append(mk_train)

            for chunk in test_chunks:
                if feature in monotonic_metric_space:
                    space = monotonic_metric_space[feature]
                    mk_test = monotoncity_metric(model.model, idx, 0, chunk, mon_increasing=increasing, k_space=space)
                else:
                    mk_test = monotoncity_metric(model.model, idx, 200, chunk, mon_increasing=increasing)
                mk_tests.append(mk_test)

            inc_dec = 'inc' if monotonic_increasing else 'dec'
            mks['mk_%s_test_%s' % (inc_dec, feature)] = statistics.mean(mk_tests)
            mks['mk_%s_train_%s' % (inc_dec, feature)] = statistics.mean(mk_trains)

    # store model
    model.store_model(path=model_log_path)
    return mae, mks


def fitness(X, y, X_test, y_test, model_creator, k_cv, log_path: str):
    # create log folder if it not exists
    dir_path = os.path.dirname('%s/mock.file' % log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pbar = tqdm.tqdm(total=len(repetition_seeds))
    # get data
    log_df = pandas.DataFrame(columns=['repetition', 'cv_split', 'inner_repetition', 'mae', 'mape'])
    for rep in range(len(repetition_seeds)):
        kf = KFold(n_splits=k_cv, shuffle=True, random_state=repetition_seeds[rep])

        # prepare folds
        process_params = []
        split = 1
        for train, test in kf.split(X):
            split_log_path = "%s/train_%d_%d" % (log_path, rep, split) + "_%d"
            model_log_path = "%s/model_%d_%d" % (log_path, rep, split) + "_%d"

            X_train = pandas.DataFrame(X.iloc[train]).reset_index(drop=True)
            y_train = pandas.DataFrame(y.iloc[train]).reset_index(drop=True)

            # validation sets for early stopping of training
            X_val = pandas.DataFrame(X.iloc[test]).reset_index(drop=True)
            y_val = pandas.DataFrame(y.iloc[test]).reset_index(drop=True)
            for inner_rep in range(INNER_REPS):
                process_params.append([
                    X_train, y_train, X_val, y_val, X_test, y_test, model_creator, split_log_path % inner_rep,
                                                                                   model_log_path % inner_rep
                ])
            split += 1

        results = Parallel(n_jobs=N_JOBS)(delayed(eval_fold)(*process_param) for process_param in process_params)

        for split in range(k_cv):
            for inner_rep in range(INNER_REPS):
                index = split * INNER_REPS + inner_rep
                (mae, mks) = results[index]
                # write log
                row = {'repetition': rep,
                       'cv_split': split,
                       'inner_repetition': inner_rep,
                       'mae': mae,
                       **mks}
                log_df = log_df.append(row, ignore_index=True)
                log_df.to_csv('%s/res.csv' % log_path, index=False)
        pbar.update(1)

    pbar.close()
    return log_df


def eval_interval(X_train, X_test, y_train, y_test, k_cv, log_path, params):
    # transform train beforehand to save execution time
    interval_transformer = IntervalTargetTransformer()
    X_train_np, y_train_np = interval_transformer.transform(X_train, y_train)
    X_train_df = pandas.DataFrame(X_train_np)
    y_train_df = pandas.DataFrame(y_train_np)

    def model_creator():
        model = IntervalRegressorMAE()
        model = model.set_params(**params['reg'])
        return model

    fitness(X_train_df, y_train_df, X_test, y_test, model_creator, k_cv, log_path % (k_cv, 'interval'))


def eval_baseline_mean(X_train, X_test, y_train, y_test, k_cv, log_path, params):
    mean_transformer = MeanTransformer()
    X_train_np, y_train_np = mean_transformer.transform(X_train, y_train)
    X_train_df = pandas.DataFrame(X_train_np)
    y_train_df = pandas.DataFrame(y_train_np)

    def model_creator():
        model = DNNRegressor()
        model = model.set_params(**params['reg'])
        return model

    fitness(X_train_df, y_train_df, X_test, y_test, model_creator, k_cv, log_path % (k_cv, 'mean'))


def eval_baseline_mean_mono(X_train, X_test, y_train, y_test, k_cv, log_path, params):
    mean_transformer = MeanTransformer()
    # monotonic increasing features
    X_train_np, y_train_np = mean_transformer.transform(X_train, y_train)
    X_train_df = pandas.DataFrame(X_train_np)
    y_train_df = pandas.DataFrame(y_train_np)

    MonotonicBatchDNNRegressor.mon_increasing = monotonic_increasing_indices
    MonotonicBatchDNNRegressor.mon_decreasing = monotonic_decreasing_indices

    def model_creator():
        MonotonicBatchDNNRegressor.mon_increasing = monotonic_increasing_indices
        MonotonicBatchDNNRegressor.mon_decreasing = monotonic_decreasing_indices
        model = MonotonicBatchDNNRegressor()
        model = model.set_params(**params['reg'])
        return model

    fitness(X_train_df, y_train_df, X_test, y_test, model_creator, k_cv, log_path % (k_cv, 'mean_mono'))


def eval_interval_mono(X_train, X_test, y_train, y_test, k_cv, log_path, params):
    mean_transformer = IntervalTargetTransformer()

    X_train_np, y_train_np = mean_transformer.transform(X_train, y_train)
    X_train_df = pandas.DataFrame(X_train_np)
    y_train_df = pandas.DataFrame(y_train_np)

    def model_creator():
        MonotonicBatchDNNRegressor.mon_increasing = monotonic_increasing_indices
        MonotonicBatchDNNRegressor.mon_decreasing = monotonic_decreasing_indices
        model = IntervalRegressorMAEMonotonic()
        model = model.set_params(**params['reg'])
        return model

    fitness(X_train_df, y_train_df, X_test, y_test, model_creator, k_cv, log_path % (k_cv, 'interval_mono'))


def full_evaluation_run(X_train, X_test, y_train, y_test, k_cv, log_path, num_datasets, params):
    if EVAL_INTERVAL:
        print("EVAL_INTERVAL")
        eval_interval(X_train, X_test, y_train, y_test, k_cv, log_path, params['interval'])

    if EVAL_BASELINE_MEAN:
        print("EVAL_BASELINE_MEAN")
        eval_baseline_mean(X_train, X_test, y_train, y_test, k_cv, log_path, params['mean'])

    if EVAL_INTERVAL_MONO:
        print("EVAL_INTERVAL_MONO")
        eval_interval_mono(X_train, X_test, y_train, y_test, k_cv, log_path, params['interval_monotonic'])

    if EVAL_BASELINE_MEAN_MONO:
        print("EVAL_BASELINE_MEAN_MONO")
        eval_baseline_mean_mono(X_train, X_test, y_train, y_test, k_cv, log_path, params['mean_monotonic'])



def evaluate_case_study(dataset_path, log_path, params):
    # train
    X_train = pandas.read_csv(dataset_path % "X_train.csv")
    y_train = pandas.read_csv(dataset_path % "y_train.csv")
    # test
    X_test = pandas.read_csv(dataset_path % "X_test.csv")
    y_test = pandas.read_csv(dataset_path % "y_test.csv")

    # convert to (len,1) shaped array
    y_test = y_test.mean(axis=1)
    # fill nans with zero if there are any
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    # get number of datasets, set to 1 for experiments, that is true for scenario 1
    # in sceneario 2 there are 2 datasets but for the second one the parameters of the second dataset are not used
    num_datasets = 1


    # shuffle train
    X_train = X_train.sample(n=SAMPLE_SIZE, random_state=27021996)
    y_train = y_train.reindex(X_train.index)

    # reset indices of all sets
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    X_test = X_test.sample(n=SAMPLE_SIZE_TEST, random_state=3011996)
    y_test = y_test.reindex(X_test.index)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # monotonic increasing features
    global monotonic_increasing_indices
    global monotonic_decreasing_indices
    monotonic_increasing_indices = [X_train.columns.get_loc(name) for name in monotonic_increasing]
    monotonic_decreasing_indices = [X_train.columns.get_loc(name) for name in monotonic_decreasing]

    global monotonic_indices_dict
    monotonic_indices_dict = dict(zip(monotonic_increasing + monotonic_decreasing,
                                      monotonic_increasing_indices + monotonic_decreasing_indices))

    # dispatch evaluation
    # full 5-fold cross validation evaluation run
    if CV_5:
        full_evaluation_run(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, k_cv=5, log_path=log_path,
                            num_datasets=num_datasets, params=params)


if __name__ == '__main__':
    # SCENARIO 1
    # read in param dicts
    with open('optimal_params/scenario_1.json', 'r') as json_file:
        params_scenario_1 = json.load(json_file)
    with open('optimal_params/scenario_2.json', 'r') as json_file:
        params_scenario_2 = json.load(json_file)

    if SCENARIO_1:
        print("SCENARIO_1")
        dataset_path_scenario_1 = "../data/case_study/scenario_1/%s"
        log_path_scenario_1 = "log/scenario_1/cv_%d/%s"
        evaluate_case_study(dataset_path=dataset_path_scenario_1, log_path=log_path_scenario_1,
                            params=params_scenario_1)

    # SCENARIO 2
    if SCENARIO_2:
        print("SCENARIO_2")
        dataset_path_scenario_2 = "../data/case_study/scenario_2/%s"
        log_path_scenario_2 = "log/scenario_2/cv_%d/%s"
        evaluate_case_study(dataset_path=dataset_path_scenario_2, log_path=log_path_scenario_2,
                            params=params_scenario_2)