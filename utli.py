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