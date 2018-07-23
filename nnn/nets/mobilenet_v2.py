# coding: utf-8

"""
Usage:
https://github.com/ornew/mobilenet/README.md

This program was implemented with reference to the following papers.

> Sandler, Mark, et al. "Inverted Residuals and Linear Bottlenecks:
> Mobile Networks for Classification, Detection and Segmentation."
> arXiv preprint arXiv:1801.04381 (2018).
"""

import tensorflow as tf

from nnn.layers import conv2d, expanded_conv2d

def _multiple(value, multipler, divisor):
  assert multipler > 0
  assert divisor > 0
  return max(divisor, (value * multipler + divisor / 2) // divisor * divisor)

def dilation_stride(output_stride, stride=1, dilation_rate=1, current_strides=1):
  assert output_stride >= current_strides
  if output_stride == current_strides:
    return (1, dilation_rate * stride, current_strides)
  else:
    return (stride, 1, current_strides * stride)

def mobilenet_v2(inputs, output_stride=16, depth_multiplier=1, expantion_rate=6, scope=None):
  with tf.variable_scope(scope, 'mobilenet_v2', [inputs]):
    x = inputs
    S, D, C = (2, 1, 2) # 1st: {stride: 2, dilation: 1} and the current stride is 2.
    x = conv2d(x, 32, 3, S, D)
    for i in [
      dict(filter=16, stride=1),
      dict(filter=24, stride=2),
      dict(filter=24, stride=1),
      dict(filter=32, stride=2),
      dict(filter=32, stride=1),
      dict(filter=32, stride=1),
      dict(filter=64, stride=2),
      dict(filter=64, stride=1),
      dict(filter=64, stride=1),
      dict(filter=64, stride=1),
      dict(filter=96, stride=1),
      dict(filter=96, stride=1),
      dict(filter=96, stride=1),
      dict(filter=160, stride=2),
      dict(filter=160, stride=1),
      dict(filter=160, stride=1),
      dict(filter=320, stride=1),
    ]:
      S, D, C = dilation_stride(output_stride, i['stride'], D, C)
      x = expanded_conv2d(
        x, i['filter'], 3, S, D, depth_multiplier, expantion_rate)
    S, D, C = dilation_stride(output_stride, 1, D, C)
    x = conv2d(x, 1280, 1, S, D)
    return x
