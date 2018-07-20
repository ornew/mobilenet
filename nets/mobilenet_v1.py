
def mobilenet_v1(inputs,
                 output_stride,
                 depth_multiplier,
                 filter_multiplier,
                 training,
                 scope=None):
  with tf.variable_scope(scope, 'mobilenet_v1', [inputs]):
    x = inputs
    x = tf.layers.conv2d(x, int(32 * filter_multiplier), 3, 2, 'same')
    x = batch_normalization(x, training=training)
    x = tf.nn.relu6(x)
    separable_conv2d_layers = [
      dict(stride=1, filter=64),
      dict(stride=2, filter=128),
      dict(stride=1, filter=128),
      dict(stride=2, filter=256),
      dict(stride=1, filter=256),
      dict(stride=2, filter=512),
      dict(stride=1, filter=512),
      dict(stride=1, filter=512),
      dict(stride=1, filter=512),
      dict(stride=1, filter=512),
      dict(stride=1, filter=512),
      dict(stride=2, filter=1024),
      dict(stride=1, filter=1024),
    ]
    strides = 1
    dilation_rate = 1
    current_strides = 1
    for i, layer in enumerate(separable_conv2d_layers):
      if output_stride == current_strides:
        strides = 1
      else:
        strides = layer['stride']
        dilation_rate = 1
        current_strides = current_strides * layer['stride']
      x = separable_conv2d_layer(x,
        filters=int(layer['filter'] * filter_multiplier),
        strides=strides,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        training=training,
        scope='separable_conv2d_{}'.format(i + 1))
      dilation_rate = dilation_rate * layer['stride']
    return tf.identity(x, name='feature_vector')
