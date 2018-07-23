# coding: utf-8

import tensorflow as tf

def image_level_features(inputs, filters, image_shape, training, scope=None):
  with tf.variable_scope(scope, 'image_level_features', [inputs]):
    x = global_average_pooling(inputs)
    x = pointwise_conv2d(x, filters, training=training)
    x = tf.image.resize_images(x, image_shape, align_corners=True)
  return x

def atrous_spatial_pyramid_pooling(inputs, filters, atrous_rates, training, scope=None):
  with tf.variable_scope(scope, 'atrous_spatial_pyramid_pooling', [inputs]):
    x = inputs
    inputs_shape = tf.shape(inputs)[1:3]
    a = pointwise_conv2d(x, filters, training=training)
    b = [
      separable_conv2d_layer(x,
        filters, kernel_size=(3,3), strides=1,
        training=training, dilation_rate=rate)
      for rate in atrous_rates
    ]
    c = image_level_features(x, filters, inputs_shape, training=training)
    x = tf.concat([a] + b + [c], axis=3)
    x = pointwise_conv2d(x, filters, training=training)
    return x
ASSP = atrous_spatial_pyramid_pooling

def deeplab_v3p(features, num_classes, atrous_rates, training, scope=None):
  regularizer = tf.contrib.layers.l2_regularizer(5e-4)
  initializer = tf.truncated_normal_initializer(stddev=1e-2)

  with tf.variable_scope(scope, 'deeplab_v3p', [features],
         regularizer=regularizer, initializer=initializer):
    with tf.variable_scope('encoder'):
      x = ASSP(features, 256, atrous_rates=atrous_rates, training=training)
      z = pointwise_conv2d(features, 48, training=training)
    with tf.variable_scope('decoder'):
      x = tf.image.resize_images(x, tf.shape(z)[1:3], align_corners=True)
      x = tf.concat([x, z], axis=3)
      x = separable_conv2d_layer(x, 256, (3,3), training=training)
      x = separable_conv2d_layer(x, 256, (3,3), training=training)
      x = tf.layers.conv2d(x, num_classes, 1, 1, 'same')
      return x
