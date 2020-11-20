import os
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.models import model_from_json

from model.callbacks import MonotonicCallback, \
    MonotonicBatchCallback
from model.loss_functions import monotonic_loss_creator, monotonic_loss_creator_batch, gen_loss_mae


class DNNRegressor(tf.keras.wrappers.scikit_learn.KerasRegressor):
    """
    A simple neural net regressor based on the KerasRegressor API. The only used layer is a fully connected dense layer.
    The activation function is relu. The following architecture parameters can be specified:
    1. num_input_nodes: number of neurons inside the first hidden layer
    2. num_dense_layers: number of hidden layers (except the first one), the created neural net contains
        num_dense_layers + 1 fully connected dense layers
    3. num_dense_nodes: number of neurons in the hidden layers
    4. ratio_dropout: dropout probability
    5. batch_size: batch_size used during training
    5. loss_func: Loss function used during training
    """

    def __init__(self, verbose: int = 0,
                 max_epochs: int = 1000,
                 batch_size: int = 64,
                 loss_func=tf.keras.losses.MeanAbsoluteError(),
                 num_dense_layers: int = 2,
                 num_input_nodes: int = 128,
                 num_dense_nodes: int = 64,
                 ratio_dropout: float = 0.15,
                 **sk_params):
        super().__init__(**sk_params)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.ratio_dropout = ratio_dropout
        self.num_dense_nodes = num_dense_nodes
        self.num_input_nodes = num_input_nodes
        self.num_dense_layers = num_dense_layers

        self.activation = 'relu'
        self.loss_func = loss_func
        self.input_dim = None

    def __call__(self, *args, **kwargs):
        """
        Creates the internally used TensorFlow2 neural net. Is used by the superclass when the fir method is called
        @param args: nothing
        @param kwargs: nothing
        @return: TensorFlow2 neural net model
        """
        # create model
        # input layer
        inp = Input(self.input_dim)
        x = Dense(self.num_input_nodes, activation=self.activation, name='input_layer')(
            inp)
        x = Dropout(self.ratio_dropout)(x)
        # hidden layer
        for i in range(self.num_dense_layers):
            name = 'layer_dense_{0}'.format(i + 1)
            x = Dense(self.num_dense_nodes, activation=self.activation, name=name)(x)
            x = Dropout(self.ratio_dropout)(x)
        # output layer
        out = Dense(1, name='output_layer')(x)

        model = Model(inputs=inp, outputs=out)
        # setup our optimizer and compile
        model.compile(optimizer='adam', loss=self.loss_func)
        return model

    def fit(self, x, y, **kwargs) -> History:
        """
        Creates the internal neural net with the parameters given by set_params() or the constructor. Afterwards trains
        the neural using X and Y. If no explicit validation set is provided (by X_val and y_val) 20% of the training set
        will be used for early stopping.
        @param x: numpy array of instances for training
        @param y: numpy array of targets for training
        @param kwargs: params are forwarded to the TensorFlow2 Api. The following parameters are intercepted:
            verbose: verbosity level for training
            epochs: maximum number of epochs
            batch_size: batch size used for training
            X_val: explicit validation set used for early stopping, if it is there while x and y is used for gradient descent
            y_val: explicit validation set used for early stopping, if it is there while x and y is used for gradient descent
            callbacks: list of callbacks passed to the TensorFlow2 moel
            log_path: path to a folder where the training log will be stored
        @return: TensorFlow2 history of training
        """
        # set input dim according to dim of x
        self.input_dim = x.shape[1]
        # set params for training
        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose
        if 'epochs' not in kwargs:
            kwargs['epochs'] = self.max_epochs
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size

        # early stopping is no_val == False
        if ('no_val' in kwargs and not kwargs['no_val']) or 'no_val' not in kwargs:
            if 'X_val' in kwargs and 'y_val' in kwargs:
                kwargs['validation_data'] = (kwargs['X_val'], kwargs['y_val'])
                kwargs.pop('X_val', None)
                kwargs.pop('y_val', None)

            else:
                kwargs['validation_split'] = 0.2

        if 'no_val' in kwargs:
            kwargs.pop('no_val', None)

        kwargs['callbacks'] = (kwargs['callbacks'] if 'callbacks' in kwargs.keys() else []) + [
            EarlyStopping(monitor='val_loss',
                          patience=50,
                          restore_best_weights=True)]

        if 'log_path' in kwargs.keys():
            kwargs['callbacks'] = [CSVLogger(filename=kwargs['log_path'])] + kwargs['callbacks']
            kwargs.pop('log_path', None)
        # delegate fit with params to super class
        return super().fit(x, y, **kwargs)

    def score(self, x, y, **kwargs):
        """
        Computes the score of the model on the test x,y. Does only make sense if the model was fitted before using the
        fit method
        @param x: instances to score
        @param y: ground truth label of instances
        @param kwargs: parameters passed to the interal used TensorFlow2 model
        @return: score of model on (x,y)
        """
        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose
        return super().score(x, y, **kwargs)

    def get_params(self, **params):
        """
        Returns the current used parameters of the DNN regressor
        @param params: nothing
        @return: dict of current parameters
        """
        return {
            'ratio_dropout': self.ratio_dropout,
            'num_dense_nodes': self.num_dense_nodes,
            'num_input_nodes': self.num_input_nodes,
            'num_dense_layers': self.num_dense_layers,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'verbose': self.verbose
        }

    def set_params(self, **parameters):
        """
        Set the parameters for the model. The parameters are:
            ratio_dropout: dropout ratio
            num_dense_nodes: number of neurons per hidden layer
            num_input_nodes: number of neurons in the first hidden layer
            num_dense_layers: number of hidden layers (net consists out of the first layer specified by num_input_nodes
                                                    + number of dense layers many layers, i.e. if num_dense_layers =1
                                                    then the net has in total 2 dense layers!
            batch_size: batch size used during training
            max_epochs: maximum number of epochs in training
            verbose: verbosity level of the TensorFlow2 model
        @param parameters: dict of parameters
        @return: self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def store_model(self, path: str):
        """
        Stores the model architecutre and the current weights at the given path
        @param path: path where the model is stored
        @return:
        """
        # serialize model
        json = self.model.to_json()
        with(open(path + '.json', "w")) as file:
            file.write(json)
        # serialize weights
        self.model.save_weights(path + '.h5')

    def load_model(self, path: str):
        """
        Loads architecture and weights of the model stored at path. Does not sets the interal params. Hence after the
        loading only predict and score should be used! If a fit is called the old parameters (before the model was
        loaded) will be used to create the model
        @param path:
        @return:
        """
        # load model from json and load weights from h5 file
        with (open(path + '.json', 'r')) as file:
            json_model = file.read()
            self.model = model_from_json(json_model)
            self.model.load_weights(path + '.h5')


class MonotonicDNNRegressor(tf.keras.wrappers.scikit_learn.KerasRegressor):
    """
    The montonic batch neural net regressor based on the KerasRegressor API. The only used layer is a fully connected dense layer.
    The monotic penalty term is computed on the WHOLE training and test set. The activation function is relu.
    The following architecture parameters can be specified:
    1. num_input_nodes: number of neurons inside the first hidden layer
    2. num_dense_layers: number of hidden layers (except the first one), the created neural net contains
        num_dense_layers + 1 fully connected dense layers
    3. num_dense_nodes: number of neurons in the hidden layers
    4. ratio_dropout: dropout probability
    5. batch_size: batch_size used during training
    5. loss_func: Loss function used during training
    6. mon_increasing: indices of features (starting at 0) that are monotone increasing
    7. mon_decreasing: indices of features (starting at 0) that are monotone decreasing
    8. penalty_weight: weight of the monotonic penalty term
    """
    mon_increasing = None
    mon_decreasing = None

    def __init__(self, penalty_weight: float = 0.5, verbose: int = 0, max_epochs: int = 1000,
                 batch_size: int = 16,
                 loss_func=tf.keras.losses.MeanAbsoluteError(), num_dense_layers: int = 2,
                 num_input_nodes: int = 128,
                 num_dense_nodes: int = 64, ratio_dropout: float = 0.15,
                 mon_increasing=None, mon_decreasing=None, **sk_params):
        super().__init__(**sk_params)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.ratio_dropout = ratio_dropout
        self.num_dense_nodes = num_dense_nodes
        self.num_input_nodes = num_input_nodes
        self.num_dense_layers = num_dense_layers

        self.activation = 'relu'
        self.loss_func = loss_func

        self.input_dim = None

        # mon increasing and mon decreasing features
        if mon_increasing is None:
            self.mon_increasing = MonotonicDNNRegressor.mon_increasing
        else:
            self.mon_increasing = mon_increasing
        if mon_decreasing is None:
            self.mon_decreasing = MonotonicDNNRegressor.mon_decreasing
        else:
            self.mon_decreasing = mon_decreasing

        # create monotonic loss func
        self.inner_loss_func = loss_func
        self.penalty_term_variable = tf.Variable(0.0, dtype=tf.float32)
        self.penalty_weight = penalty_weight

    def __call__(self, *args, **kwargs):
        """
        Creates the internally used TensorFlow2 neural net. Is used by the superclass when the fir method is called
        @param args: nothing
        @param kwargs: nothing
        @return: TensorFlow2 neural net model
        """
        # create model
        # input layer
        inp = Input(self.input_dim)
        x = Dense(self.num_input_nodes, activation=self.activation, name='input_layer')(
            inp)
        x = Dropout(self.ratio_dropout)(x)
        # hidden layer
        for i in range(self.num_dense_layers):
            name = 'layer_dense_{0}'.format(i + 1)
            x = Dense(self.num_dense_nodes, activation=self.activation, name=name)(x)
            x = Dropout(self.ratio_dropout)(x)
        # output layer
        out = Dense(1, name='output_layer')(x)

        model = Model(inputs=inp, outputs=out)

        self.mon_increasing_mask = [index in self.mon_increasing for index in range(self.input_dim)]
        self.mon_decreasing_mask = [index in self.mon_decreasing for index in range(self.input_dim)]

        self.loss_func = monotonic_loss_creator(self.penalty_weight,
                                                self.inner_loss_func,
                                                self.X_train_tensor,
                                                self.X_test_tensor,
                                                model,
                                                self.mon_increasing_mask, self.mon_decreasing_mask,
                                                self.train_indicator,
                                                self.last_penalty)
        # setup our optimizer and compile
        model.compile(optimizer='adam', loss=self.loss_func)
        return model

    def fit(self, x, y, **kwargs) -> History:
        """
        Creates the internal monotonic neural net with the parameters given by set_params() or the constructor. Afterwards trains
        the neural using X and Y. If no explicit validation set is provided (by X_val and y_val) 20% of the training set
        will be used for early stopping. In this version the whole training and validation set used to compute the monotonic
        penalty term.
        @param x: numpy array of instances for training
        @param y: numpy array of targets for training
        @param kwargs: params are forwarded to the TensorFlow2 Api. The following parameters are intercepted:
                verbose: verbosity level for training
                epochs: maximum number of epochs
                batch_size: batch size used for training
                X_val: explicit validation set used for early stopping, if it is there while x and y is used for gradient descent
                y_val: explicit validation set used for early stopping, if it is there while x and y is used for gradient descent
                callbacks: list of callbacks passed to the TensorFlow2 moel
                log_path: path to a folder where the training log will be stored
        @return: TensorFlow2 history of training
        """
        self.input_dim = x.shape[1]
        self.train_indicator = tf.Variable(0)
        self.last_penalty = tf.Variable(1.0)

        if 'X_val' not in kwargs and 'y_val' not in kwargs:
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)
        else:
            X_train = x
            y_train = y
            X_val = kwargs['X_val']
            y_val = kwargs['y_val']
            # pop args from dict
            kwargs.pop('X_val')
            kwargs.pop('y_val')

        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose
        if 'epochs' not in kwargs:
            kwargs['epochs'] = self.max_epochs
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        # validation split by hand to feed X_train and X_test into MonotonicPenaltyCallback

        if 'log_path' in kwargs.keys():
            log_path = kwargs['log_path']
            kwargs.pop('log_path', None)
        else:
            log_path = None

        # add callback
        kwargs['callbacks'] = [MonotonicCallback(self.train_indicator, self.last_penalty),
                               EarlyStopping(monitor='val_loss',
                                             patience=50,
                                             restore_best_weights=True)]

        self.X_train_tensor = tf.convert_to_tensor(X_train)
        self.X_test_tensor = tf.convert_to_tensor(X_val)

        # add callback
        hist = super().fit(x=X_train, y=y_train, validation_data=(X_val, y_val), **kwargs)
        return hist

    def score(self, x, y, **kwargs):
        """
        Computes the score of the model on the test x,y. Does only make sense if the model was fitted before using the
        fit method
        @param x: instances to score
        @param y: ground truth label of instances
        @param kwargs: parameters passed to the interal used TensorFlow2 model
        @return: score of model on (x,y)
        """
        return -mean_absolute_error(self.predict(x), y)

    def get_params(self, **params):
        """
        Returns the current used parameters of the monotonic DNN regressor
        @param params: nothing
        @return: dict of current parameters
        """
        return {
            'ratio_dropout': self.ratio_dropout,
            'num_dense_nodes': self.num_dense_nodes,
            'num_input_nodes': self.num_input_nodes,
            'num_dense_layers': self.num_dense_layers,
            'penalty_weight': self.penalty_weight,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'verbose': self.verbose
        }

    def set_params(self, **parameters):
        """
        Set the parameters for the model. The parameters are:
        ratio_dropout: dropout ratio
        num_dense_nodes: number of neurons per hidden layer
        num_input_nodes: number of neurons in the first hidden layer
        num_dense_layers: number of hidden layers (net consists out of the first layer specified by num_input_nodes
                                                    + number of dense layers many layers, i.e. if num_dense_layers =1
                                                    then the net has in total 2 dense layers!
        batch_size: batch size used during training
        penalty_weight: weight of the monotonic penalty term
        max_epochs: maximum number of epochs in training
        verbose: verbosity level of the TensorFlow2 model
        @param parameters: dict of parameters
        @return: self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def store_model(self, path: str):
        """
        Stores the model architecutre and the current weights at the given path
        @param path: path where the model is stored
        @return:
        """
        # serialize model
        json = self.model.to_json()
        with(open(path + '.json', "w")) as file:
            file.write(json)
        # serialize weights
        self.model.save_weights(path + '.h5')

    def load_model(self, path: str):
        """
        Loads architecture and weights of the model stored at path. Does not sets the interal params. Hence after the
        loading only predict and score should be used! If a fit is called the old parameters (before the model was
        loaded) will be used to create the model
        @param path:
        @return:
        """
        # load model from json and load weights from h5 file
        with (open(path + '.json', 'r')) as file:
            json_model = file.read()
            self.model = model_from_json(json_model)
            self.model.load_weights(path + '.h5')


class MonotonicBatchDNNRegressor(tf.keras.wrappers.scikit_learn.KerasRegressor):
    """
    The montonic batch neural net regressor based on the KerasRegressor API. The only used layer is a fully connected dense layer.
    The monotic penalty term is computed on the current BATCH. The activation function is relu.
    The following architecture parameters can be specified:
    1. num_input_nodes: number of neurons inside the first hidden layer
    2. num_dense_layers: number of hidden layers (except the first one), the created neural net contains
        num_dense_layers + 1 fully connected dense layers
    3. num_dense_nodes: number of neurons in the hidden layers
    4. ratio_dropout: dropout probability
    5. batch_size: batch_size used during training
    5. loss_func: Loss function used during training
    6. mon_increasing: indices of features (starting at 0) that are monotone increasing
    7. mon_decreasing: indices of features (starting at 0) that are monotone decreasing
    8. penalty_weight: weight of the monotonic penalty term
    """

    mon_increasing = None
    mon_decreasing = None

    def __init__(self, penalty_weight: float = 0.5, verbose: int = 0, max_epochs: int = 1000,
                 batch_size: int = 64,
                 loss_func=tf.keras.losses.MeanAbsoluteError(), num_dense_layers: int = 2,
                 num_input_nodes: int = 128,
                 num_dense_nodes: int = 64, ratio_dropout: float = 0.15,
                 mon_increasing=None, mon_decreasing=None, **sk_params):
        super().__init__(**sk_params)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.ratio_dropout = ratio_dropout
        self.num_dense_nodes = num_dense_nodes
        self.num_input_nodes = num_input_nodes
        self.num_dense_layers = num_dense_layers

        self.activation = 'relu'
        self.loss_func = loss_func

        self.input_dim = None

        # mon increasing and mon decreasing features
        if mon_increasing is None:
            self.mon_increasing = MonotonicBatchDNNRegressor.mon_increasing
        else:
            self.mon_increasing = mon_increasing
        if mon_decreasing is None:
            self.mon_decreasing = MonotonicBatchDNNRegressor.mon_decreasing
        else:
            self.mon_decreasing = mon_decreasing

        # create monotonic loss func
        self.inner_loss_func = loss_func
        self.penalty_term_variable = tf.Variable(0.0, dtype=tf.float32)
        self.penalty_weight = penalty_weight

    def __call__(self, *args, **kwargs):
        """
        Creates the internally used TensorFlow2 neural net. Is used by the superclass when the fir method is called
        @param args: nothing
        @param kwargs: nothing
        @return: TensorFlow2 neural net model
        """
        # create model
        # input layer
        inp = Input(self.input_dim)
        x = Dense(self.num_input_nodes, activation=self.activation, name='input_layer')(
            inp)
        x = Dropout(self.ratio_dropout)(x)
        # hidden layer
        for i in range(self.num_dense_layers):
            name = 'layer_dense_{0}'.format(i + 1)
            x = Dense(self.num_dense_nodes, activation=self.activation, name=name)(x)
            x = Dropout(self.ratio_dropout)(x)
        # output layer
        out = Dense(1, name='output_layer')(x)

        model = Model(inputs=inp, outputs=out)

        self.mon_increasing_mask = [index in self.mon_increasing for index in range(self.input_dim)]
        self.mon_decreasing_mask = [index in self.mon_decreasing for index in range(self.input_dim)]

        self.loss_func = monotonic_loss_creator_batch(self.penalty_weight,
                                                      self.inner_loss_func,
                                                      self.X_train_batches,
                                                      self.X_train_batches,
                                                      model,
                                                      self.mon_increasing_mask, self.mon_decreasing_mask,
                                                      self.train_indicator,
                                                      self.last_penalty,
                                                      self.current_step)
        # setup our optimizer and compile
        model.compile(optimizer='adam', loss=self.loss_func)
        return model

    def fit(self, x, y, **kwargs) -> History:
        """
        Creates the internal monotonic neural net with the parameters given by set_params() or the constructor. Afterwards trains
        the neural using X and Y. If no explicit validation set is provided (by X_val and y_val) 20% of the training set
        will be used for early stopping. In this version the monotonic penalty term is only computed on the current batch.
        @param x: numpy array of instances for training
        @param y: numpy array of targets for training
        @param kwargs: params are forwarded to the TensorFlow2 Api. The following parameters are intercepted:
                verbose: verbosity level for training
                epochs: maximum number of epochs
                batch_size: batch size used for training
                X_val: explicit validation set used for early stopping, if it is there while x and y is used for gradient descent
                y_val: explicit validation set used for early stopping, if it is there while x and y is used for gradient descent
                callbacks: list of callbacks passed to the TensorFlow2 moel
                log_path: path to a folder where the training log will be stored
        @return: TensorFlow2 history of training
               """
        self.input_dim = x.shape[1]
        self.train_indicator = tf.Variable(0)
        self.last_penalty = tf.Variable(1.0)
        self.current_step = tf.Variable(0, dtype=tf.int32)

        if 'X_val' not in kwargs and 'y_val' not in kwargs:
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)
        else:
            X_train = x
            y_train = y
            X_val = kwargs['X_val']
            y_val = kwargs['y_val']
            # pop args from dict
            kwargs.pop('X_val')
            kwargs.pop('y_val')

        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose
        if 'epochs' not in kwargs:
            kwargs['epochs'] = self.max_epochs
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        # validation split by hand to feed X_train and X_test into MonotonicPenaltyCallback

        if 'log_path' in kwargs.keys():
            log_path = kwargs['log_path']
            kwargs.pop('log_path', None)
        else:
            log_path = None

        # use own batching for monotonic penaly
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(y)).batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = test_dataset.shuffle(buffer_size=len(y)).batch(self.batch_size)

        train_batches = [tf.convert_to_tensor(batch[0]) for batch in train_dataset.as_numpy_iterator()]
        test_batches = [tf.convert_to_tensor(batch[0]) for batch in test_dataset.as_numpy_iterator()]

        # modify last tensor to fit the batch size
        length = tf.shape(train_batches[-1]).numpy()[0]
        repetition_count = self.batch_size // length
        repetitions = numpy.ones(length, dtype=numpy.int) * repetition_count
        to_less = self.batch_size - repetitions.sum()
        repetitions[0:to_less] = repetitions[0:to_less] + 1
        train_batches[-1] = tf.repeat(train_batches[-1], repetitions, axis=0)

        length = tf.shape(test_batches[-1]).numpy()[0]
        repetition_count = self.batch_size // length
        repetitions = numpy.ones(length, dtype=numpy.int) * repetition_count
        to_less = self.batch_size - repetitions.sum()
        repetitions[0:to_less] = repetitions[0:to_less] + 1
        test_batches[-1] = tf.repeat(test_batches[-1], repetitions, axis=0)

        self.X_train_batches = tf.stack(train_batches)
        self.X_test_batches = tf.stack(test_batches)

        # add callback
        kwargs['callbacks'] = [MonotonicBatchCallback(self.train_indicator, self.last_penalty, self.current_step),
                               EarlyStopping(monitor='val_loss',
                                             patience=50,
                                             restore_best_weights=True)]

        # add callback
        return super().fit(x=train_dataset, y=None, validation_data=test_dataset, **kwargs)

    def score(self, x, y, **kwargs):
        """
        Computes the score of the model on the test x,y. Does only make sense if the model was fitted before using the
        fit method
        @param x: instances to score
        @param y: ground truth label of instances
        @param kwargs: parameters passed to the interal used TensorFlow2 model
        @return: score of model on (x,y)
        """
        pred = self.predict(x).reshape(-1, 1)
        return -self.inner_loss_func(y, pred).numpy()

    def get_params(self, **params):
        """
        Returns the current used parameters of the monotonic DNN regressor
        @param params: nothing
        @return: dict of current parameters
        """
        return {
            'ratio_dropout': self.ratio_dropout,
            'num_dense_nodes': self.num_dense_nodes,
            'num_input_nodes': self.num_input_nodes,
            'num_dense_layers': self.num_dense_layers,
            'penalty_weight': self.penalty_weight,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size
        }

    def set_params(self, **parameters):
        """
        Set the parameters for the model. The parameters are:
        ratio_dropout: dropout ratio
        num_dense_nodes: number of neurons per hidden layer
        num_input_nodes: number of neurons in the first hidden layer
        num_dense_layers: number of hidden layers
        batch_size: batch size used during training
        penalty_weight: weight of the monotonic penalty term
        max_epochs: maximum number of epochs in training
        verbose: verbosity level of the TensorFlow2 model
        @param parameters: dict of parameters
        @return: self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def store_model(self, path: str):
        """
        Stores the model architecutre and the current weights at the given path
        @param path: path where the model is stored
        @return:
        """
        # serialize model
        json = self.model.to_json()
        with(open(path + '.json', "w")) as file:
            file.write(json)
        # serialize weights
        self.model.save_weights(path + '.h5')

    def load_model(self, path: str):
        """
        Loads architecture and weights of the model stored at path. Does not sets the interal params. Hence after the
        loading only predict and score should be used! If a fit is called the old parameters (before the model was
        loaded) will be used to create the model
        @param path:
        @return:
        """
        # load model from json and load weights from h5 file
        with (open(path + '.json', 'r')) as file:
            json_model = file.read()
            self.model = model_from_json(json_model)
            self.model.load_weights(path + '.h5')



class IntervalRegressorMAE(DNNRegressor):

    def __init__(self, verbose: int = 0,
                 max_epochs: int = 1000,
                 batch_size: int = 64,
                 num_dense_layers: int = 2,
                 num_input_nodes: int = 128,
                 num_dense_nodes: int = 64,
                 ratio_dropout: float = 0.15,
                 **sk_params):
        super().__init__(
            verbose,
            max_epochs,
            batch_size,
            gen_loss_mae,
            num_dense_layers,
            num_input_nodes,
            num_dense_nodes,
            ratio_dropout,
            **sk_params)


class IntervalRegressorMAEMonotonic(MonotonicBatchDNNRegressor):

    def __init__(self,
                 penalty_weight: float = 0.5,
                 verbose: int = 0,
                 max_epochs: int = 1000,
                 batch_size: int = 64,
                 num_dense_layers: int = 2,
                 num_input_nodes: int = 128,
                 num_dense_nodes: int = 64,
                 ratio_dropout: float = 0.15,
                 **sk_params):
        super().__init__(
            penalty_weight,
            verbose,
            max_epochs,
            batch_size,
            gen_loss_mae,
            num_dense_layers,
            num_input_nodes,
            num_dense_nodes,
            ratio_dropout,
            **sk_params)
