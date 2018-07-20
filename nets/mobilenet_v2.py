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

from layers import conv2d, expanded_conv2d

Conv = Convolution = namedtuple(
  'Convolution',
  ['kernel_size', 'filters', 'strides'])
ExpSepConv = ExpandedSeparableConvolution = namedtuple(
  'ExpandedSeparableConvolution',
  ['expantion_rate', 'filters', 'strides'])

MOBILENET_V2_LAYERS = [
  Conv      (3,   32, 2),
  ExpSepConv(1,   16, 1),
  ExpSepConv(6,   24, 2),
  ExpSepConv(6,   24, 1),
  ExpSepConv(6,   32, 2),
  ExpSepConv(6,   32, 1),
  ExpSepConv(6,   32, 1),
  ExpSepConv(6,   64, 2),
  ExpSepConv(6,   64, 1),
  ExpSepConv(6,   64, 1),
  ExpSepConv(6,   64, 1),
  ExpSepConv(6,   96, 1),
  ExpSepConv(6,   96, 1),
  ExpSepConv(6,   96, 1),
  ExpSepConv(6,  160, 2),
  ExpSepConv(6,  160, 1),
  ExpSepConv(6,  160, 1),
  ExpSepConv(6,  320, 1),
  Conv      (1, 1280, 1),
]

def _multiple(value, multipler, divisor):
  assert multipler > 0
  assert divisor > 0
  return max(divisor, (value * multipler + divisor / 2) // divisor * divisor)

def mobilenet_v2(inputs, multiplier=None, scope=None):
  with tf.variable_scope(scope, 'mobilenet_v2', [inputs]):
    conv2d()

def mobilenet_v2(inputs, multiplier=None, scope=None):
  with tf.variable_scope(scope, 'mobilenet_v2', [inputs]):
    x = inputs
    for i, l in enumerate(MOBILENET_V2_LAYERS):
      with tf.variable_scope('hidden_layer_{}'.format(i)):
        num_outputs = _multiple(l.filters, multiplier, 8) \
          if multiplier is not None else l.filters

        if isinstance(l, Convolution):
          x = tf.layers.conv2d(
              x, num_outputs, l.kernel_size, l.strides, 'SAME', use_bias=False)
          x = tf.layers.batch_normalization(x, training=is_training)
          x = tf.nn.relu6(x)

        elif isinstance(l, ExpandedSeparableConvolution):
          x = expanded_separable_convolution2d(
              x, num_outputs, l.strides, l.expantion_rate, is_training)

      tf.logging.debug('mobilenet_v2.hidden_layer.{:02}: output_shape={}'.format(
          i, x.get_shape().as_list()))
  return x
