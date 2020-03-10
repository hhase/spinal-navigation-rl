import tensorflow as tf
import numpy as np
import time
import pickle
import pdb

def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        print("   [-] %s : %2.5f sec, which is %2.5f mins, which is %2.5f hours" %
              (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed

def _debug(operation):
    print("Layer_name: " + operation.op.name + " -Output_Shape: " + str(operation.shape.as_list()))

# Summaries for variables
def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    #variable_summaries(w)
    return w


def _residual_block(name, x, filters, pool_first=False, strides=1, dilation=1, bias=-1):
    print('Building residual unit: %s' % name)
    with tf.variable_scope(name):
        # get input channels
        in_channel = x.shape.as_list()[-1]

        # Shortcut connection
        shortcut = tf.identity(x)

        if pool_first:
            if in_channel == filters:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut= tf.pad(x, tf.constant([[0,0],[1,1],[1,1],[0,0]]), "CONSTANT")
                    shortcut = tf.nn.max_pool(shortcut, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = _conv('shortcut_conv', x, padding='VALID',
                                      num_filters=filters, kernel_size=(1, 1), stride=(strides, strides),
                                      bias=bias)
        else:
            if dilation != 1:
                shortcut = _conv('shortcut_conv', x, padding='VALID',
                                      num_filters=filters, kernel_size=(1, 1), dilation=dilation, bias=bias)

        # Residual
        x = _conv('conv_1', x, padding=[[0,0],[1,1],[1,1],[0,0]],
                       num_filters=filters, kernel_size=(3, 3), stride=(strides, strides), bias=bias)
        #x = _bn('bn_1', x)
        x = _relu('relu_1', x)
        x = _conv('conv_2', x, padding=[[0,0],[1,1],[1,1],[0,0]],
                       num_filters=filters, kernel_size=(3, 3), bias=bias)
        #x = _bn('bn_2', x)

        # Merge
        x = x + shortcut
        x = _relu('relu_2', x)

        print('residual-unit-%s-shape: ' % name + str(x.shape.as_list()))

        return x

def _conv(name, x, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
          initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, dilation=1.0, bias=-1):

    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)

        #variable_summaries(w)
        if dilation > 1:
            conv = tf.nn.atrous_conv2d(x, w, dilation, padding)
        else:
            if type(padding)==type(''):
                conv = tf.nn.conv2d(x, w, stride, padding)
            else:
                conv = tf.pad(x, padding, "CONSTANT")
                conv = tf.nn.conv2d(conv, w, stride, padding='VALID')

        if bias != -1:
            bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))

            #variable_summaries(bias)
            conv = tf.nn.bias_add(conv, bias)

        tf.add_to_collection('debug_layers', conv)

        return conv

def _relu(name, x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

def _fc(name, x, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=-1):

    with tf.variable_scope(name):
        n_in = x.get_shape()[-1].value

        w = variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)

        #variable_summaries(w)

        if bias != -1 and isinstance(bias, float):
            bias = tf.get_variable("biases", [output_dim], tf.float32, tf.constant_initializer(bias))
            output = tf.nn.bias_add(tf.matmul(x, w), bias)
        else:
            output = tf.matmul(x, w)

        return output

def _bn(name, x, train_flag):
    with tf.variable_scope(name):
        moving_average_decay = 0.9
        decay = moving_average_decay

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

        mu = tf.get_variable('mu', batch_mean.shape, dtype=tf.float32,
                             initializer=tf.zeros_initializer(), trainable=False)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, mu)
        tf.add_to_collection('mu_sigma_bn', mu)
        sigma = tf.get_variable('sigma', batch_var.shape, dtype=tf.float32,
                                initializer=tf.ones_initializer(), trainable=False)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, sigma)
        tf.add_to_collection('mu_sigma_bn', sigma)
        beta = tf.get_variable('beta', batch_mean.shape, dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', batch_var.shape, dtype=tf.float32,
                                initializer=tf.ones_initializer())

        # BN when training
        update = 1.0 - decay
        update_mu = mu.assign_sub(update * (mu - batch_mean))
        update_sigma = sigma.assign_sub(update * (sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(train_flag, lambda: (batch_mean, batch_var), lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        tf.add_to_collection('debug_layers', bn)

        return bn


def ResNet18(x_input=None, classes=5, bias=-1, weight_decay=5e-4, test_classification=False):

    with tf.variable_scope('conv1_x'):
        print('Building unit: conv1')
        conv1 = _conv('conv1', x_input, padding= [[0,0],[3,3],[3,3],[0,0]],
                                num_filters=64, kernel_size=(7, 7), stride=(2, 2), l2_strength=weight_decay,
                                bias=bias)

        #conv1 = _bn('bn1', conv1)

        conv1 = _relu('relu1', conv1)
        _debug(conv1)
        conv1= tf.pad(conv1, tf.constant([[0,0],[1,1],[1,1],[0,0]]), "CONSTANT")
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                    name='max_pool1')
        _debug(conv1)
        print('conv1-shape: ' + str(conv1.shape.as_list()))

    with tf.variable_scope('conv2_x'):
        conv2 = _residual_block('conv2_1', conv1, 64)
        _debug(conv2)
        conv2 = _residual_block('conv2_2', conv2, 64)
        _debug(conv2)

    with tf.variable_scope('conv3_x'):
        conv3 = _residual_block('conv3_1', conv2, 128, pool_first=True, strides=2)
        _debug(conv3)
        conv3 = _residual_block('conv3_2', conv3, 128)
        _debug(conv3)

    with tf.variable_scope('conv4_x'):
        conv4 = _residual_block('conv4_1', conv3, 256, pool_first=True, strides=2)
        _debug(conv4)
        conv4 = _residual_block('conv4_2', conv4, 256)
        _debug(conv4)

    with tf.variable_scope('conv5_x'):
        conv5 = _residual_block('conv5_1', conv4, 512, pool_first=True, strides=2)
        _debug(conv5)
        conv5 = _residual_block('conv5_2', conv5, 512)
        _debug(conv5)

    with tf.variable_scope('resnet_out'):
        print('Building unit: logits')
        #score = tf.reduce_mean(conv5, axis=[1, 2])
        score = tf.compat.v1.layers.flatten(conv5)
        _debug(score)
        score = _fc('logits_dense', score, output_dim=classes, l2_strength=weight_decay)
        print('logits-shape: ' + str(score.shape.as_list()))

    return score