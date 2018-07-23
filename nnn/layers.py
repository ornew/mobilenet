# coding: utf-8

import tensorflow as tf
from nnn.arg_scope import add_args
from nnn.util import get_pair, get_channel, get_strides, get_conv2d_weights

@add_args
def batch_normalization(
      inputs,
      training,
      *args, **kwargs):
  return tf.layers.batch_normalization(
    inputs,
    training=training,
    *args, **kwargs)

@add_args
def add_bias(inputs, name='biases'):
  channel = get_channel(inputs)
  b = tf.get_variable(name, [channel])
  x = tf.nn.bias_add(inputs, b)
  return x

@add_args
def depthwise_conv2d(
    inputs,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    depth_multiplier=1,
    use_bias=True,
    normalizer=None,
    activation=tf.nn.relu6,
    scope=None):
  with tf.variable_scope(scope, 'depthwise_conv2d', [inputs]):
    x = inputs
    W = get_conv2d_weights(x, depth_multiplier, kernel_size)
    S = get_strides(strides)
    D = get_pair(dilation_rate)
    x = tf.nn.depthwise_conv2d(x, W, S, 'SAME', rate=D)
    x = add_bias(x) if use_bias else x
    x = normalizer(x) if normalizer else x
    x = activation(x) if activation else x
    return x

@add_args
def conv2d(
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    use_bias=True,
    normalizer=None,
    activation=tf.nn.relu6,
    scope=None):
  with tf.variable_scope(scope, 'conv2d', [inputs]):
    x = inputs
    W = get_conv2d_weights(x, filters, kernel_size)
    S = get_strides(strides)
    D = get_pair(dilation_rate)
    D = [1, D[0], D[1], 1]
    x = tf.nn.conv2d(x, W, S, 'SAME', dilations=D)
    x = add_bias(x) if use_bias else x
    x = normalizer(x) if normalizer else x
    x = activation(x) if activation else x
    return x

@add_args
def pointwise_conv2d(
    inputs,
    filters,
    use_bias=True,
    normalizer=None,
    activation=tf.nn.relu6,
    scope=None):
  with tf.variable_scope(scope, 'pointwise_conv2d', [inputs]):
    x = inputs
    W = get_conv2d_weights(x, filters, 1)
    S = get_strides(1)
    x = tf.nn.conv2d(inputs, W, S, 'SAME')
    x = add_bias(x) if use_bias else x
    x = normalizer(x) if normalizer else x
    x = activation(x) if activation else x
    return x

@add_args
def separable_conv2d(
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    depth_multiplier=1,
    use_bias=True,
    normalizer=None,
    activation=tf.nn.relu6,
    scope=None):
  with tf.variable_scope(scope, 'separable_conv2d', [inputs]):
    x = inputs
    x = depthwise_conv2d(x, kernel_size, strides, dilation_rate,
        depth_multiplier, use_bias, normalizer, activation)
    x = pointwise_conv2d(x, filters, use_bias, normalizer, activation)
    return x

@add_args
def expanded_conv2d(
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    depth_multiplier=1,
    expantion_rate=6,
    use_bias=True,
    normalizer=None,
    activation=tf.nn.relu6,
    residual=True):
  """Expanded Convolution-2d Layer"""
  with tf.variable_scope(None, 'expanded_conv2d', [inputs]):
    x = inputs
    assert expantion_rate >= 1
    channel = get_channel(inputs)
    expand_filters = int(expantion_rate * channel)
    x = pointwise_conv2d(x, expand_filters,
                         normalizer=normalizer, activation=activation,
                         scope='expand') if expantion_rate > 1 else x
    x = depthwise_conv2d(x, kernel_size, strides, dilation_rate,
                         depth_multiplier, use_bias=use_bias, normalizer=normalizer,
                         activation=activation, scope='depthwise')
    x = pointwise_conv2d(x, filters, use_bias=use_bias, normalizer=normalizer,
                         activation=None, scope='project')

    # residual connection.
    if callable(residual):
      x = residual(inputs, x)
    elif residual and strides == (1,1) and get_channel(inputs) == get_channel(x):
      x = tf.add(inputs, x)
    return x

@add_args
def global_average_pooling(inputs, scope=None):
  with tf.variable_scope(scope, 'global_average_pooling', [inputs]):
    return tf.reduce_mean(inputs, [1, 2], keepdims=True, name='value')
