import tensorflow as tf
import tensorflow.keras.backend as kb


def gen_loss_mae(y_actual, y_pred):
    """
    Generalized loss function for the absolute error.
    @param y_actual: tensor containing ground truth values
    @param y_pred: tensor containing the lower and upper interval boundaries
    @return:
    """
    # get lower and upper interval boundary
    y_actual_lower = y_actual[:, 0]
    y_actual_upper = y_actual[:, 1]
    y_pred = kb.flatten(y_pred)
    # use chained max operator with zero tensor to compute loss
    zero = kb.zeros_like(y_pred, dtype=tf.float32)
    t1 = kb.maximum(zero, y_actual_lower - y_pred)
    t2 = kb.maximum(zero, y_pred - y_actual_upper)
    return kb.mean(kb.maximum(t1, t2))


def gen_loss_mse(y_actual, y_pred):
    """
    Generalized loss function for the squared error.
    @param y_actual: tensor containing ground truth values
    @param y_pred: tensor containing the lower and upper interval boundaries
    @return:
    """
    # get lower and upper interval boundary
    y_actual_lower = y_actual[:, 0]
    y_actual_upper = y_actual[:, 1]
    y_pred = kb.flatten(y_pred)
    # use chained max operator with zero tensor to compute loss
    zero = kb.zeros_like(y_pred, dtype=tf.float32)
    t1 = kb.maximum(zero, y_actual_lower - y_pred)
    t2 = kb.maximum(zero, y_pred - y_actual_upper)
    return kb.mean(kb.square(kb.maximum(t1, t2)))


def divergence_of_tensor(X_tensor: tf.Tensor, model, mon_increasing_mask, mon_decreasing_mask):
    """
    Computes the divergence of model model with respect to X_tensor for features indices that are 1 in
    mon_increasing_mask and mon_decreasing_mask
    @param X_tensor: TensorFlow2 tensor of input elements
    @param model: model to compute the divergence for
    @param mon_increasing_mask: boolean mask that indicates which features should be incorporated into the computation
        for the monotone increasing portion of the penalty term
    @param mon_decreasing_mask: boolean mask that indicates which features should be incorporated into the computation
        for the monotone decreasing portion of the penalty term
    @return: tensor that holds the value of the monotonic penalty term
    """
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        batch_pred = model(X_tensor)
    gradients = tape.gradient(batch_pred, X_tensor)

    mon_incr_gradients = tf.boolean_mask(gradients, mon_increasing_mask, axis=1)
    mon_decr_gradients = tf.boolean_mask(gradients, mon_decreasing_mask, axis=1)

    mon_incr_divergence = -tf.math.reduce_sum(mon_incr_gradients, axis=1)
    mon_decr_divergence = tf.math.reduce_sum(mon_decr_gradients, axis=1)

    mon_incr_sum = tf.math.reduce_sum(tf.math.maximum(0.0, mon_incr_divergence))
    mon_decr_sum = tf.math.reduce_sum(tf.math.maximum(0.0, mon_decr_divergence))

    mon_sum = mon_incr_sum + mon_decr_sum
    return mon_sum


def monotonic_loss_creator(penalty_weight, inner_loss, X_train: tf.Tensor, X_test: tf.Tensor, model, mon_increasing_mask, mon_decreasing_mask,
                             train_indicator: tf.Variable, last_penalty: tf.Variable):
    """
    Wrapper to create a loss augmented with the monotonic penalty term. The FULL training and validation set is used
    for the computation of the penalty term.
    @param penalty_weight: weight of the penalty term
    @param inner_loss: loss that will be augmented
    @param X_train: X portion of the training set
    @param X_test: X portion of the validation set
    @param model: model in which the loss function will be used for training
    @param mon_increasing_mask: boolean mask that indicates which features should be incorporated into the computation
        for the monotone increasing portion of the penalty term
    @param mon_decreasing_mask: boolean mask that indicates which features should be incorporated into the computation
        for the monotone decreasing portion of the penalty term
    @param train_indicator: tensorflow variable that indicates if the model is in train or test mode
    @param last_penalty: tensorflow to store the value of the penalty term (only for logging)
    @return: inner_loss augmented with penalty term
    """

    def monotonic_loss(y_actual, y_pred):
        is_train = tf.constant(1)

        def f1():
            return penalty_weight * divergence_of_tensor(X_train, model, mon_increasing_mask, mon_decreasing_mask)

        def f2():
            return penalty_weight * divergence_of_tensor(X_test, model, mon_increasing_mask, mon_decreasing_mask)

        penalty = tf.cond(tf.equal(is_train, train_indicator), f1, f2)
        last_penalty.assign(penalty)
        return inner_loss(y_actual, y_pred) + penalty
    return monotonic_loss


def monotonic_loss_creator_batch(penalty_weight, inner_loss, X_train_batches: tf.Tensor , X_test_batches: tf.Tensor, model, mon_increasing_mask, mon_decreasing_mask,
                             train_indicator: tf.Variable, last_penalty: tf.Variable, current_step: tf.Variable):

    """
    Wrapper to create a loss augmented with the monotonic penalty term. The current BATCH of training and validation set
    is used for the computation of the penalty term.
    @param penalty_weight: weight of the penalty term
    @param inner_loss: loss that will be augmented
    @param X_train: X portion of the training set
    @param X_test: X portion of the validation set
    @param model: model in which the loss function will be used for training
    @param mon_increasing_mask: boolean mask that indicates which features should be incorporated into the computation
        for the monotone increasing portion of the penalty term
    @param mon_decreasing_mask: boolean mask that indicates which features should be incorporated into the computation
        for the monotone decreasing portion of the penalty term
    @param train_indicator: tensorflow variable that indicates if the model is in train or test mode
    @param last_penalty: tensorflow variabe to store the value of the penalty term (only for logging)
    @param current_step: tensorflow variabe that indicates which batch of training/test set is processed at the moment
    @return:
    """
    def monotonic_loss(y_actual, y_pred):
        is_train = tf.constant(1)

        def f1():
            train_batch = X_train_batches[current_step.read_value()]
            return penalty_weight * divergence_of_tensor(train_batch, model, mon_increasing_mask, mon_decreasing_mask)

        def f2():
            test_batch = X_test_batches[current_step.read_value()]
            return penalty_weight * divergence_of_tensor(test_batch, model, mon_increasing_mask, mon_decreasing_mask)

        penalty = tf.cond(tf.equal(is_train, train_indicator), f1, f2)
        last_penalty.assign(penalty)
        return inner_loss(y_actual, y_pred) + penalty
    return monotonic_loss
