import tensorflow as tf
from math import sqrt


# Scale and shift parameters in this layer
def scale_shift(c_i, name):
    shape = c_i.get_shape()[-1:]
    sh = tf.get_variable(name=name + '_shift', shape=shape, dtype=tf.float32, trainable=True,
                         initializer=tf.zeros_initializer)
    sc = tf.get_variable(name=name + '_scale', shape=shape, dtype=tf.float32, trainable=True,
                         initializer=tf.zeros_initializer)
    sc += 1.0
    return c_i * sc + sh


def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters
