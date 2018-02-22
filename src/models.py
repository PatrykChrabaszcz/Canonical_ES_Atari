from tensorflow.contrib.layers import conv2d as conv
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import batch_norm as bn
from .ops import scale_shift
import tensorflow as tf
import numpy as np

# Some configuration parameters for convolution and dense layers
# This will avoid unnecessary repeats
# Be careful when you change those values as many models will overwrite
# some fields
conv_args = {
    "padding": "VALID",
    "biases_initializer": None,
    "activation_fn": None,
    "weights_initializer": tf.random_normal_initializer(0, 0.05)
}
dense_args = {
    "biases_initializer": None,
    "activation_fn": None,
    "weights_initializer": tf.random_normal_initializer(0, 0.05)
}

bn_args = {
    "decay": 0.,
    "center": True,
    "scale": False,
    "epsilon": 1e-8,
    "activation_fn": tf.nn.elu,
    "is_training": False
}


# Assumes that this will be called only once and changes bn_args dictionary
def Nature(c_i, out_num, nonlin, is_training):
    # Adjust bn_args dictionary
    bn_args["activation_fn"] = nonlin
    bn_args["is_training"] = is_training
    c_i = conv(c_i, 32, 8, 4, **conv_args)
    c_i = bn(c_i, **bn_args)

    c_i = conv(c_i, 64, 4, 2, **conv_args)
    c_i = bn(c_i, **bn_args)

    c_i = conv(c_i, 64, 3, 1, **conv_args)
    c_i = bn(c_i, **bn_args)

    c_i = tf.reshape(c_i, [-1, np.prod([int(s) for s in c_i.get_shape()[1:]])])
    c_i = fc(c_i, num_outputs=512, **dense_args)
    c_i = bn(c_i, **bn_args)

    # We can't use "center" here. We have to apply scale first and then shift
    # This will be done with "scale_shift" layer
    bn_args["center"] = False
    bn_args["activation_fn"] = None
    c_i = fc(c_i, num_outputs=out_num, **dense_args)
    c_i = bn(c_i, **bn_args)

    # We can't add scale to this batch norm because we might want to apply L2 reg and we would like
    # this scale to be centered in 1 instead of 0
    c_i = scale_shift(c_i, name="scale")
    return c_i
