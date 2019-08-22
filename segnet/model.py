import tensorflow as tf
from layers import max_pool, variable_with_weights_decay, conv2d, up_sampling, initialization


def segnet(inputs, batch_size, num_classes, is_training):
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
    # 1st encode block
    conv1_1 = conv2d(norm1, shape=[3, 3, 3, 64], is_training=is_training, name='conv1_1')
    conv1_2 = conv2d(conv1_1, shape=[3, 3, 64, 64], is_training=is_training, name='conv1_2')
    pool1 = max_pool(conv1_2, name='pool1')
    
    #2th block
    conv2_1 = conv2d(pool1, shape=[3, 3, 64, 128], is_training=is_training, name='conv2_1')
    conv2_2 = conv2d(conv2_1, shape=[3, 3, 128, 128], is_training=is_training, name='conv2_2')
    pool2 = max_pool(conv2_2, name='pool2')
    
    #3th block
    conv3_1 = conv2d(pool2, shape=[3, 3, 128, 256], is_training=is_training, name='conv3_1')
    conv3_2 = conv2d(conv3_1, shape=[3, 3, 256, 256], is_training=is_training, name='conv3_2')
    conv3_3 = conv2d(conv3_2, shape=[3, 3, 256, 256], is_training=is_training, name='conv3_3')
    pool3 = max_pool(conv3_3, name='pool3')
    
    #4th block
    conv4_1 = conv2d(pool3, shape=[3, 3, 256, 512], is_training=is_training, name='conv4_1')
    conv4_2 = conv2d(conv4_1, shape=[3, 3, 512, 512], is_training=is_training, name='conv4_2')
    conv4_3 = conv2d(conv4_2, shape=[3, 3, 512, 512], is_training=is_training, name='conv4_3')
    pool4 = max_pool(conv4_3, name='pool4')
    
    #5th block
    conv5_1 = conv2d(pool4, shape=[3, 3, 512, 512], is_training=is_training, name='conv5_1')
    conv5_2 = conv2d(conv5_1, shape=[3, 3, 512, 512], is_training=is_training, name='conv5_2')
    conv5_3 = conv2d(conv5_2, shape=[3, 3, 512, 512], is_training=is_training, name='conv5_3')
    pool5 = max_pool(conv5_3, name='pool5')
    
    ### decode block
    #1st deconv block
    unpool5 = up_sampling(pool5, name='unpool5')
    deconv5_3 = conv2d(unpool5, shape=[3, 3, 512, 512], is_training=is_training, name='deconv5_3')
    deconv5_2 = conv2d(deconv5_3, shape=[3, 3, 512, 512], is_training=is_training, name='deconv5_2')
    deconv5_1 = conv2d(deconv5_2, shape=[3, 3, 512, 512], is_training=is_training, name='deconv5_1')
    
    #2ed deconv block
    unpool4 = up_sampling(deconv5_1, name='unpool4')
    deconv4_3 = conv2d(unpool4, shape=[3, 3, 512, 512], is_training=is_training, name='deconv4_3')
    deconv4_2 = conv2d(deconv4_3, shape=[3, 3, 512, 512], is_training=is_training, name='deconv4_2')
    deconv4_1 = conv2d(deconv4_2, shape=[3, 3, 512, 256], is_training=is_training, name='deconv4_1')
    
    #3th deconv block
    unpool3 = up_sampling(deconv4_1, name='unpool3')
    deconv3_3 = conv2d(unpool3, shape=[3, 3, 256, 256], is_training=is_training, name='deconv3_3')
    deconv3_2 = conv2d(deconv3_3, shape=[3, 3, 256, 256], is_training=is_training, name='deconv3_2')
    deconv3_1 = conv2d(deconv3_2, shape=[3, 3, 256, 128], is_training=is_training, name='deconv3_1')

    #4th deconv block
    unpool2 = up_sampling(deconv3_1, name='unpool2')
    deconv2_2 = conv2d(unpool2, shape=[3, 3, 128, 128], is_training=is_training, name='deconv2_2')
    deconv2_1 = conv2d(deconv2_2, shape=[3, 3, 128, 64], is_training=is_training, name='deconv2_1')
    
    #4th deconv block
    unpool1 = up_sampling(deconv2_1, name='unpool1')
    deconv1_2 = conv2d(unpool1, shape=[3, 3, 64, 64], is_training=is_training, name='deconv1_2')
    deconv1_1 = conv2d(deconv1_2, shape=[3, 3, 64, 64], is_training=is_training, name='deconv1_1')
    
    #Start classifiy
    with tf.variable_scope('conv_classifier') as scope:
        kernel = variable_with_weights_decay([1, 1, 64, num_classes], 
                                             initializer=initialization(1, 64), 
                                             wd=0.0005, 
                                             name='weights')
        conv = tf.nn.conv2d(deconv1_1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = variable_with_weights_decay([num_classes], 
                                             initializer=tf.constant_initializer(0.0), 
                                             wd=False, 
                                             name='biases')
        logits = tf.nn.bias_add(conv, biases, name=scope.name)
        
    return logits
        

    
 