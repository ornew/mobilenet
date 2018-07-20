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

def depthwise_conv2d(
    inputs,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    depth_multiplier=1,
    normalizer=None,
    activation=tf.nn.relu6,
    use_bias=True,
    scope=None):
  assert training is not None
  dilation_rate = get_pair(dilation_rate)
  kernel_size = get_pair(kernel_size)
  in_channel = get_channel(inputs)
  with tf.variable_scope(scope, 'depthwise_conv2d', [inputs]):
    W = tf.get_variable('weights',
        [kernel_size[0], kernel_size[1], in_channel, depth_multiplier],
        regularizer=tf.no_regularizer,
        initializer=kernel_initializer)
    x = tf.nn.depthwise_conv2d(inputs, W, strides, 'SAME', rate=dilation_rate)
    if use_bias:
      b = tf.get_variable('biases',
          [in_channel * depth_multiplier],
          initializer=bias_initializer)
      x = tf.nn.bias_add(x, b)
    x = normalizer(x) if normalizer else x
    x = activation(x) if activation else x
    return x

def pointwise_conv2d(inputs, filters, training, scope=None):
  with tf.variable_scope(scope, 'pointwise_conv2d', [inputs]):
    W = tf.get_variable('weight',
        [1, 1, depth_output_channel, filters],
        initializer=kernel_initializer)
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
    if use_bias:
      b = tf.get_variable('biases',
          [in_channel * depth_multiplier],
          initializer=bias_initializer)
      x = tf.nn.bias_add(x, b)
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
    training=None,
    depthwise_kernel_initializer=tf.glorot_uniform_initializer(),
    depthwise_bias_initializer=tf.zeros_initializer(),
    pointwise_kernel_initializer=tf.glorot_uniform_initializer(),
    pointwise_bias_initializer=tf.zeros_initializer(),
    scope=None):
  
  depth_output_channel = int(in_channel * depth_multiplier)
  with tf.variable_scope(scope, 'separable_conv2d'):
    x = depthwise_conv2d(x)
    x = pointwise_conv2d(x)
    return x

def expanded_conv2d(inputs, filters, stride, expantion_rate, is_training):
  """Expanded Separable Convolution2d Layer"""
  with tf.variable_scope(None, 'expanded_conv2d', [inputs]):
    x = inputs
    assert expantion_rate >= 1
    channel = _get_channel(inputs)
    num_outputs = int(expantion_rate * channel)
    if expantion_rate > 1:
        x = _expansion_conv2d_layer(x, expantion_rate, is_training, activation=tf.nn.relu6)
    x = _depthwise_conv2d_layer(x, stride,  is_training, activation=tf.nn.relu6)
    x = _pointwise_conv2d_layer(x, filters, is_training, activation=None)

    if stride == 1 and _get_channel(inputs) == _get_channel(x):
        x = tf.add(inputs, x) # residual connection.
    return x

def global_average_pooling(inputs, scope=None):
  with tf.variable_scope(scope, 'global_average_pooling', [inputs]):
    return tf.reduce_mean(inputs, [1, 2], keepdims=True, name='value')
