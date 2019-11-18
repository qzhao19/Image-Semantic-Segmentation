

import tensorflow as tf
from layers import max_pool, conv2d, up_sampling, fc


def fcn16(inputs, num_classes, is_training):
    """segnet model function
    
    Args:
        inputs: inputs tensor with 4D dimensions 
        batch_size:
        is_training:
    Returns:
        logits value
    """
    
    norm1 = tf.nn.lrn(inputs, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
    ### encode block
    conv1_1 = conv2d(x, shape=[3, 3, 3, 64], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv1_1')
    conv1_2 = conv2d(conv1_1, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv1_2')
    pool1 = max_pooling(conv1_2, ksize=[2, 2], strides=[2, 2], padding='SAME', name='pool1')

    # second conv block
    conv2_1 = conv2d(pool1, shape=[3, 3, 64, 128], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv2_1')
    conv2_2 = conv2d(conv2_1, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv2_2')
    pool2 = max_pooling(conv2_2, ksize=[2, 2], strides=[2, 2], padding='SAME', name='pool2')

    # 3th conv block
    conv3_1 = conv2d(pool2, shape=[3, 3, 128, 256], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv3_1')
    conv3_2 = conv2d(conv3_1, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv3_2')
    conv3_3 = conv2d(conv3_2, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv3_3')
    pool3 = max_pooling(conv3_3, ksize=[2, 2], strides=[2, 2], padding='SAME', name='pool3')

    # 4th conv block
    conv4_1 = conv2d(pool3, shape=[3, 3, 256, 512], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv4_1')
    conv4_2 = conv2d(conv4_1, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv4_2')
    conv4_3 = conv2d(conv4_2, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv4_3')
    pool4 = max_pooling(conv4_3, ksize=[2, 2], strides=[2, 2], padding='SAME', name='pool4')

    # 5th conv block
    conv5_1 = conv2d(pool4, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv5_1')
    conv5_2 = conv2d(conv5_1, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv5_2')
    conv5_3 = conv2d(conv5_2, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', is_training=is_training, name='conv5_3')
    pool5 = max_pooling(conv5_3, ksize=[2, 2], strides=[2, 2], padding='SAME', name='pool5')


    flatten_shape = pool5.get_shape()[1].value * pool5.get_shape()[2].value * pool5.get_shape()[3].value
    fc6 = tf.reshape(pool5, shape=[-1, flatten_shape])

    fc6 = fc(fc6, shape=[flatten_shape, 4096], name='fc1')
    if is_training:
        fc6 = dropout(fc6, keep_prob=0.5, name='dropout1')

    fc7 = fc(fc6, shape=[4096, 4096], name='fc2')
    if is_training:
        fc7 = dropout(fc7, keep_prob=0.5, name='dropout2')

    fc8 = fc(fc7, shape=[4096, 1000], name='fc3')



        
    return logits
