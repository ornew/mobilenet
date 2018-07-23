# coding: utf-8

import tensorflow as tf

def get_pair(x):
  if type(x) == tuple or type(x) == list:
    if len(x) == 2:
      return tuple(x)
    else:
      raise ValueError('The tuple element count is must be 2.')
  elif type(x) == int or type(x) == long:
    return int(x),int(x)
  else:
    raise ValueError('The type for input is must be tuple, list, or int.')

def get_channel(x, data_format='channel_last'):
  assert x.shape.ndims == 4
  if data_format is 'channel_last':
    return x.shape[-1]
  elif data_format is 'channel_first':
    return x.shape[1]

def get_strides(strides):
  strides = get_pair(strides)
  return [1, strides[0], strides[1], 1]

def get_conv2d_weights(
    inputs, 
    filters,
    kernel_size,
    name='weights'):
  in_channel  = get_channel(inputs)
  kernel_size = get_pair(kernel_size)
  shape = [
    kernel_size[0], kernel_size[1],
    in_channel, filters
  ]
  return tf.get_variable(name, shape)
