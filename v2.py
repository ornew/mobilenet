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

Convolution = namedtuple('Convolution', ['kernel_size', 'filters', 'strides'])
Bottleneck  = namedtuple('Bottleneck',  ['expantion_rate', 'filters', 'strides'])

MOBILENET_V2_LAYERS = [
    Convolution(3,   32, 2),
    Bottleneck( 1,   16, 1),
    Bottleneck( 6,   24, 2),
    Bottleneck( 6,   24, 1),
    Bottleneck( 6,   32, 2),
    Bottleneck( 6,   32, 1),
    Bottleneck( 6,   32, 1),
    Bottleneck( 6,   64, 2),
    Bottleneck( 6,   64, 1),
    Bottleneck( 6,   64, 1),
    Bottleneck( 6,   64, 1),
    Bottleneck( 6,   96, 1),
    Bottleneck( 6,   96, 1),
    Bottleneck( 6,   96, 1),
    Bottleneck( 6,  160, 2),
    Bottleneck( 6,  160, 1),
    Bottleneck( 6,  160, 1),
    Bottleneck( 6,  320, 1),
    Convolution(1, 1280, 1),
]

def _get_channel(tensor):
    channel_axis = -1 # NHWC is -1, NCHW is 1
    channel = tensor.get_shape().as_list()[channel_axis]
    assert channel is not None
    return channel

def _expansion_conv2d_layer(inputs, expantion_rate, is_training, activation=None):
    channel = _get_channel(inputs)
    num_outputs = int(expantion_rate * channel)
    with tf.variable_scope(scope, 'expansion_conv2d_layer', [inputs]):
        x = tf.layers.conv2d(inputs, num_outputs, 1, 'same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        if activation:
            x = activation(x)
        return x

def _depthwise_conv2d_layer(inputs, stride, is_training, depth_multiplier=1, activation=None):
    channel = _get_channel(inputs)
    dtype = inputs.dtype.base_dtype
    shape = [3, 3, channel, depth_multiplier]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(scope, 'depthwise_conv2d_layer', [inputs]):
        kernel = tf.get_variable('kernel', shape=shape, dtype=dtype)
        x = tf.nn.depthwise_conv2d(
            inputs, kernel, strides, 'same', data_format='NHWC')
        x = tf.layers.batch_normalization(x, training=is_training)
        if activation:
            x = activation(x)
        return x

def _pointwise_conv2d_layer(inputs, filters, is_training, activation=None):
    with tf.variable_scope(scope, 'depthwise_conv2d_layer', [inputs]):
        x = tf.layers.conv2d(inputs, filters, 1, 'same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        if activation:
            x = activation(x)
        return x

def inverted_bottleneck_layer(inputs, expantion_rate, filters, stride, is_training, scope=None):
    with tf.variable_scope(scope, 'inverted_bottleneck_layer', [inputs]):
        x = inputs
        x = _expansion_conv2d_layer(x, expantion_rate, activation=tf.nn.relu6)
        x = _depthwise_conv2d_layer(x, stride        , activation=tf.nn.relu6)
        x = _pointwise_conv2d_layer(x, filters       , activation=None)
        if stride == 1:
            x = tf.add(inputs, x) # residual connection.
        return x

def mobilenet(inputs, is_training, scope=None):
    with tf.variable_scope(scope, 'mobilenet_v2', [inputs]):
        x = inputs
        for i, l in enumerate(MOBILENET_V2_LAYERS):
            layer_name = 'hidden_layer_{}'.format(i)
            with tf.variable_scope(layer_name):
                
                if isinstance(l, Convolution):
                    x = tf.layers.conv2d(
                        x, l.filters, l.kernel_size, l.strides, 'same', use_bias=False)
                    x = tf.layers.batch_normalization(x, training=is_training)
                    
                elif isinstance(l, Bottleneck):
                    x = inverted_bottleneck_layer(
                        x, l.expantion_rate, l.filters, l.strides, is_training)

            tf.logging.debug('mobilenet_v2: {}: output_shape={}'.format(
                layer_name, x.get_shape().as_list()))
        return x

def classify(inputs, num_classes, is_training, dropout_keep_prob=0.999, scope=None):
    with tf.variable_scope(scope, 'mobilenet_v2_classify', [inputs]):
        x = mobilenet(inputs, is_training)
        x = tf.reduce_mean(x, [1, 2], keepdims=True, name='global_average_pooling')
        assert x.get_shape().as_list() == [None, 1, 1, 1280]
        x = tf.layers.dropout(x, rate=dropout_keep_prob, training=is_training)
        x = tf.layers.conv2d(x, num_classes, 1, name='readout')
        x = tf.squeeze(x, [1, 2])
        assert x.get_shape().as_list() == [None, num_classes]
        return x
