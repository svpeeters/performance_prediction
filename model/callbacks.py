import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class MonotonicCallback(tf.keras.callbacks.Callback):
    """
    Callback that organizes the monotonic penalty computation for the training and th validation set. If the model
    is in training mode the penalty term incorporates the training set and otherwise the validation set. This callback
    is needed since TensorFlow 2 does not allow to have separated loss functions for training and validation set (set
    on which the early stopping will be done). This callback essentially alters a TensorFlow variable which indicates
    if the model is in training mode or not. Based on the Variable the monotonic loss function uses the training or the
    validation set for the computation of the penalty term.
    """

    def __init__(self, train_indicator: tf.Variable, last_penalty: tf.Variable):
        super().__init__()
        self.train_indicator = train_indicator
        self.last_penalty = last_penalty
        self.train_penalty = 0.0
        self.val_penalty = 0.0
        self.num_train_batches = 0
        self.num_val_batches = 0

    def on_train_begin(self, logs=None):
        self.train_indicator.assign(1)

    def on_train_batch_begin(self, batch, logs=None):
        self.num_train_batches += 1

    def on_train_batch_end(self, batch, logs=None):
        self.train_penalty += self.last_penalty.numpy()

    def on_test_begin(self, logs=None):
        self.train_indicator.assign(0)

    def on_test_batch_begin(self, batch, logs=None):
        self.num_val_batches += 1

    def on_test_end(self, logs=None):
        self.train_indicator.assign(1)

    def on_test_batch_end(self, batch, logs=None):
        self.val_penalty += self.last_penalty.numpy()

    def on_epoch_end(self, epoch, logs=None):
        logs['mon_penalty'] =  self.train_penalty / self.num_train_batches
        logs['val_mon_penalty'] = self.val_penalty / self.num_val_batches
        self.train_penalty = 0.0
        self.val_penalty = 0.0
        self.num_train_batches = 0
        self.num_val_batches = 0


class MonotonicBatchCallback(tf.keras.callbacks.Callback):
    """
    Callback that organizes the monotonic penalty computation for the training and th validation set. If the model
    is in training mode the penalty term incorporates the training set and otherwise the validation set. This callback
    is needed since TensorFlow 2 does not allow to have separated loss functions for training and validation set (set
    on which the early stopping will be done). This callback essentially alters a TensorFlow variable which indicates
    if the model is in training mode or not and which batch is the current batch used in training and validation.
    Based on the Variables the monotonic loss function uses the  corresponding batch from the training or the validation
    set for the computation of the penalty term.
    """

    def __init__(self, train_indicator: tf.Variable, last_penalty: tf.Variable, current_step):
        super().__init__()
        self.train_indicator = train_indicator
        self.last_penalty = last_penalty
        self.train_penalty = 0.0
        self.current_step = current_step
        self.val_penalty = 0.0
        self.num_train_batches = 0
        self.num_val_batches = 0

    def on_train_begin(self, logs=None):
        self.train_indicator.assign(1)

    def on_train_batch_begin(self, batch, logs=None):
        self.current_step.assign(self.num_train_batches)
        self.num_train_batches += 1

    def on_train_batch_end(self, batch, logs=None):
        self.train_penalty += self.last_penalty.numpy()

    def on_test_begin(self, logs=None):
        self.train_indicator.assign(0)

    def on_test_batch_begin(self, batch, logs=None):
        self.current_step.assign(self.num_val_batches)
        self.num_val_batches += 1

    def on_test_end(self, logs=None):
        self.train_indicator.assign(1)

    def on_test_batch_end(self, batch, logs=None):
        self.val_penalty += self.last_penalty.numpy()

    def on_epoch_end(self, epoch, logs=None):
        logs['mon_penalty'] =  self.train_penalty / self.num_train_batches
        logs['val_mon_penalty'] = self.val_penalty / self.num_val_batches
        self.train_penalty = 0.0
        self.val_penalty = 0.0
        self.num_train_batches = 0
        self.num_val_batches = 0