# coding: utf-8

import tensorflow as tf
from util import get_pair, get_channel

def batch_normalization(
      inputs,
      training,
      *args, **kwargs):
  return tf.layers.batch_normalization(
    inputs,
    training=training,
    *args, **kwargs)

def add_bias(inputs, bias_name='biases'):
  channel = get_channel(inputs)
  b = tf.get_variable(bias_name, [channel])
  x = tf.nn.bias_add(x, b)
  return x

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
  dilation_rate = get_pair(dilation_rate)
  strides = get_pair(strides)
  strides = [1,strides[0],strides[1],1]
  kernel_size = get_pair(kernel_size)
  channel = get_channel(inputs)
  with tf.variable_scope(scope, 'depthwise_conv2d', [inputs]):
    W = tf.get_variable('weights',
        [kernel_size[0], kernel_size[1], channel, depth_multiplier])
    x = tf.nn.depthwise_conv2d(inputs, W, strides, 'SAME', rate=dilation_rate)
    x = add_bias(x) if use_bias else x
    x = normalizer(x) if normalizer else x
    x = activation(x) if activation else x
    return x

def pointwise_conv2d(
    inputs,
    filters,
    use_bias=True,
    normalizer=None,
    activation=tf.nn.relu6,
    scope=None):
  channel = get_channel(inputs)
  with tf.variable_scope(scope, 'pointwise_conv2d', [inputs]):
    W = tf.get_variable('weight', [1, 1, channel, filters])
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
    x = add_bias(x) if use_bias else x
    x = normalizer(x) if normalizer else x
    x = activation(x) if activation else x
    return x

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
  x = inputs
  with tf.variable_scope(scope, 'separable_conv2d'):
    x = depthwise_conv2d(x,  kernel_size, strides, dilation_rate,
        depth_multiplier, use_bias, normalizer, activation)
    x = pointwise_conv2d(x, filters, use_bias, normalizer, activation)
    return x

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
  x = inputs
  with tf.variable_scope(None, 'expanded_conv2d', [inputs]):
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
    elif residual and stride == 1 and get_channel(inputs) == get_channel(x):
      x = tf.nn.add(inputs, x)
    return x

def global_average_pooling(inputs, scope=None):
  with tf.variable_scope(scope, 'global_average_pooling', [inputs]):
    return tf.reduce_mean(inputs, [1, 2], keepdims=True, name='value')
