import ast
import os

import numpy
import pandas
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


cpu_numerical_features = ['cpu_# of Cores', 'cpu_# of Threads', 'cpu_Frequency', 'cpu_Turbo Clock',
                          'cpu_Multiplier', 'cpu_Multiplier Unlocked',
                          'cpu_Base Clock', 'cpu_Cache L1', 'cpu_Cache L2', 'cpu_Cache L3', 'cpu_Die Size',
                          'cpu_Process Size', 'cpu_Transistors',
                          'cpu_SMP # CPUs', 'cpu_TDP']

gpu_numerical_features = ['gpu_Base Clock', 'gpu_Boost Clock', 'gpu_Bandwidth', 'gpu_Compute Units', 'gpu_Die Size',
                          'gpu_Process Size',
                          'gpu_Transistors', 'gpu_Memory Bus', 'gpu_Memory Size', 'gpu_FP32 (float) performance',
                          'gpu_Pixel Rate', 'gpu_Texture Rate', 'gpu_ROPs', 'gpu_Shading Units', 'gpu_TMUs']

num_columns = cpu_numerical_features + gpu_numerical_features + ['resolution']

def create_folder(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def exflate_target(df):
    # exflate fps_sample
    samples = []
    # exflate fps_sample
    for index, row in df.iterrows():
        if row['dataset'] == 0:
            # userbenchmark
            sample_dict = ast.literal_eval(row['fps_sample'])
            fps_sample = []
            for fps, freq in sample_dict.items():
                for _ in range(int(freq)):
                    fps_sample.append(float(fps))
            samples.append(numpy.array(fps_sample))
        else:
            samples.append(numpy.array([row['fps']]))

    y_fps = pandas.DataFrame(samples)
    return y_fps


def impute(train, test):
    if len(train.shape) == 1:
        train = numpy.reshape(train.to_numpy(), (-1, 1))
        test = numpy.reshape(test.to_numpy(), (-1, 1))
    imp_mean = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    train_imputed = imp_mean.fit_transform(train)
    test_imputed = imp_mean.transform(test)
    return train_imputed, test_imputed


def scale(train, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled


def select_features(train, test):
    columns = pandas.Series(train.columns)
    game_columns = list(columns[columns.str.startswith('game_')])

    sel = VarianceThreshold(threshold=0.9 * (1 - 0.9))
    sel.fit(train)

    sel_columns = list(columns[sel.get_support()]) + game_columns + ['dataset']

    return train[sel_columns], test[sel_columns]


def create_scenario1(in_file, out_path):
    fps = pandas.read_csv(in_file)
    fps = fps.drop(columns=['cpu_name', 'gpu_name', 'sample_count'])

    # convert settings from one hot to ordinary encoding
    setting_low = fps['setting_low']
    setting_med = fps['setting_med']
    setting_max = fps['setting_max']
    # high is base level
    setting_high = ((setting_low + setting_med + setting_max) == 0).to_numpy(dtype=int)

    setting_ordinary = setting_low * 0 + setting_med * 1 + setting_high * 2 + setting_max * 3

    fps['setting'] = setting_ordinary

    fps = fps.drop(columns=['setting_low', 'setting_med', 'setting_max'])

    # split ub and fb partition
    ub = fps[fps['dataset'] == 0.0]
    fb = fps[fps['dataset'] == 1.0]

    train = ub
    test = fb

    # select features and target
    y_train = exflate_target(train)
    y_test = exflate_target(test)

    X_train = train.drop(columns=['fps_sample', 'fps'])
    X_test = test.drop(columns=['fps_sample', 'fps'])

    # impute and standard scale numeric columns
    X_train[num_columns], X_test[num_columns] = scale(*impute(X_train[num_columns], X_test[num_columns]))

    # apply variance threshhold feature selector
    X_train, X_test = select_features(X_train, X_test)

    create_folder("%s/X_train.csv" % out_path)

    # to csv
    X_train.to_csv("%s/X_train.csv" % out_path, index=False)
    X_test.to_csv("%s/X_test.csv" % out_path, index=False)

    y_train.to_csv("%s/y_train.csv" % out_path, index=False)
    y_test.to_csv("%s/y_test.csv" % out_path, index=False)


def create_scenario2(in_file, out_path):
    fps = pandas.read_csv(in_file)
    fps = fps.drop(columns=['cpu_name', 'gpu_name', 'sample_count'])

    # convert settings from one hot to ordinary encoding
    setting_low = fps['setting_low']
    setting_med = fps['setting_med']
    setting_max = fps['setting_max']
    # high is base level
    setting_high = ((setting_low + setting_med + setting_max) == 0).to_numpy(dtype=int)

    setting_ordinary = setting_low * 0 + setting_med * 1 + setting_high * 2 + setting_max * 3

    fps['setting'] = setting_ordinary

    fps = fps.drop(columns=['setting_low', 'setting_med', 'setting_max'])

    # split ub and fb partition
    ub = fps[fps['dataset'] == 0.0]
    fb = fps[fps['dataset'] == 1.0]

    fb_train_indices = list(fb.sample(frac=0.5).index)
    fb_test_indices = [elem for elem in fb.index if elem not in fb_train_indices]

    train = pandas.concat([ub, fb.loc[fb_train_indices]])
    test = fb.loc[fb_test_indices]

    # select features and target
    y_train = exflate_target(train)
    y_test = exflate_target(test)

    X_train = train.drop(columns=['fps_sample', 'fps'])
    X_test = test.drop(columns=['fps_sample', 'fps'])

    # impute and standard scale numeric columns
    X_train[num_columns], X_test[num_columns] = scale(*impute(X_train[num_columns], X_test[num_columns]))

    # apply variance threshhold feature selector
    X_train, X_test = select_features(X_train, X_test)

    create_folder("%s/X_train.csv" % out_path)

    # to csv
    X_train.to_csv("%s/X_train.csv" % out_path, index=False)
    X_test.to_csv("%s/X_test.csv" % out_path, index=False)

    y_train.to_csv("%s/y_train.csv" % out_path, index=False)
    y_test.to_csv("%s/y_test.csv" % out_path, index=False)


if __name__ == '__main__':
    """
    Creates the scenaro 1 and 2 datasets for the case study from the full_dataset.csv file
    """
    create_scenario1("full_dataset.csv", "scenario_1_new")
    create_scenario1("full_dataset.csv", "scenario_2_new")