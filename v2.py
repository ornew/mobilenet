# coding: utf-8

"""
Usage:
https://github.com/ornew/mobilenet/README.md

This program was implemented with reference to the following papers.

> Sandler, Mark, et al. "Inverted Residuals and Linear Bottlenecks:
> Mobile Networks for Classification, Detection and Segmentation."
> arXiv preprint arXiv:1801.04381 (2018).
"""

from collections import namedtuple
import tensorflow as tf

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

def _get_channel(tensor):
    channel_axis = -1 # NHWC is -1, NCHW is 1
    channel = tensor.get_shape().as_list()[channel_axis]
    assert channel is not None
    return channel

def _multiple(value, multipler, divisor):
    assert multipler > 0
    assert divisor > 0
    return max(divisor, (value * multipler + divisor / 2) // divisor * divisor)

def _expansion_conv2d_layer(inputs, expantion_rate, is_training, activation=None):
    channel = _get_channel(inputs)
    num_outputs = int(expantion_rate * channel)
    with tf.variable_scope(None, 'expansion_conv2d_layer', [inputs]):
        x = tf.layers.conv2d(inputs, num_outputs, 1, 1, 'SAME', use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        if activation:
            x = activation(x)
        return x

def _depthwise_conv2d_layer(inputs, stride, is_training, depth_multiplier=1, activation=None):
    channel = _get_channel(inputs)
    dtype = inputs.dtype.base_dtype
    shape = [3, 3, channel, depth_multiplier]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(None, 'depthwise_conv2d_layer', [inputs]):
        kernel = tf.get_variable('kernel', shape=shape, dtype=dtype)
        x = tf.nn.depthwise_conv2d(
            inputs, kernel, strides, 'SAME', data_format='NHWC')
        x = tf.layers.batch_normalization(x, training=is_training)
        if activation:
            x = activation(x)
        return x

def _pointwise_conv2d_layer(inputs, filters, is_training, activation=None):
    with tf.variable_scope(None, 'pointwise_conv2d_layer', [inputs]):
        x = tf.layers.conv2d(inputs, filters, 1, 1, 'SAME', use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        if activation:
            x = activation(x)
        return x

def expanded_separable_convolution2d(inputs, filters, stride, expantion_rate, is_training):
    """Expanded Separable Convolution2d Layer
    """
    with tf.variable_scope(None, 'expanded_separable_convolution2d_layer', [inputs]):
        x = inputs
        assert expantion_rate >= 1
        if expantion_rate > 1:
            x = _expansion_conv2d_layer(x, expantion_rate, is_training, activation=tf.nn.relu6)
        x = _depthwise_conv2d_layer(x, stride , is_training, activation=tf.nn.relu6)
        x = _pointwise_conv2d_layer(x, filters, is_training, activation=None)

        if stride == 1 and _get_channel(inputs) == _get_channel(x):
            x = tf.add(inputs, x) # residual connection.
        return x

def mobilenet(inputs, is_training, multiplier=None, scope=None):
    """Embedding Feature by MobileNet v2
    """
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

def classify(
        inputs, num_classes, is_training, multiplier=None,
        dropout_keep_prob=0.999, scope=None):
    """MobileNet v2 Classification
    """
    with tf.variable_scope(scope, 'mobilenet_v2_classify', [inputs]):
        x = mobilenet(inputs, is_training, multiplier=multiplier)
        x = tf.reduce_mean(x, [1, 2], keepdims=True, name='global_average_pooling')
        x = tf.layers.dropout(x, rate=dropout_keep_prob, training=is_training)
        x = tf.layers.conv2d(x, num_classes, 1, name='readout')
        x = tf.squeeze(x, [1, 2])
        assert x.get_shape().as_list() == [None, num_classes]
        return x
