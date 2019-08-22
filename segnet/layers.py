import tensorflow as tf
import math

def max_pool(inputs, name):
    """max_pooling layer
    Args: 
        inputs: 4D tensor with [batch, height, width, channels]
        name
    Returns:
        A 4D tensor and output shape (inputs.get_shape().as_list())
    """
    with tf.variable_scope(name) as scope:
        value = tf.nn.max_pool(tf.cast(inputs, tf.float32), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
        
    return tf.cast(value, tf.float32)



def variable_with_weights_decay(shape, initializer, wd, name):
    """The function helps to create an initialized variable with weights decay. The variable is initialized by the trucated normalize distribution. 
    by the specific function tf.initializers.he_uniform(). 
    Args:
        shape: the shape of kernel size
        initializer: initialized variable
        wd: weights decay rate
        name: 
    
    """
    # 
    var = tf.get_variable(name, shape=shape, initializer=initializer)
    if wd is True:
        # calculate the losses value with weight decay
        # add loss value into collection by function add_to_collection
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        
    return var
        

def batch_norm(inputs, is_training, scope):
    """batch normalization layer
    
    """
    with tf.variable_scope(scope.name) as scope:
        return tf.cond(is_training, 
                       lambda: tf.contrib.layers.batch_norm(inputs, is_training=True, center=False, scope=scope), 
                       lambda: tf.contrib.layers.batch_norm(inputs, is_training=False, center=False, reuse=True, scope=scope))
    


def conv2d(inputs, shape, is_training, name):
    """convolution layer
    
    Args:
        inputs: 4D tensor. The input image and label tensor
        shape: the shape of kernel size
        is_training: training state, which represents if the weights should update 
        name: corresponding layer's name
        
    Returns:
        The output from layers
    """
    with tf.variable_scope(name) as scope:
        # get weights value and record a summary protocol buffer with a histogram.
        weights = variable_with_weights_decay(shape=shape, initializer=tf.initializers.he_normal(), wd=0.1, name='weights')
        tf.summary.histogram(scope.name + 'weight', weights)
        
        # get biases value and record a summary protocol buffer with a histogram.
        biases = variable_with_weights_decay(shape=shape[3], initializer=tf.constant_initializer(0.0), wd=False, name='biases')
        tf.summary.histogram(scope.name + 'bias', biases)
        
        # calc conv2d: image * weights
        conv_weight = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
        # calc conv2d with image * weights - biases
        conv_bias = tf.nn.bias_add(conv_weight, biases)
        conv_outputs = tf.nn.relu(batch_norm(conv_bias, is_training, scope))

    return conv_outputs
        

    
def up_sampling(pool, name):
    """Unpooling layer after max_pool
    
    Args:
       pool:   max pooled output tensor
    
    """
    with tf.variable_scope(name):
        pool_shape = tf.shape(pool)
        return tf.image.resize_images(pool, (2*pool_shape[1], 2*pool_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    
    
    
def initialization(ksize, num_channels):
    """Here we use kailing initialization, the inference paper: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    We consider for all conv layer, the kernel matrix follows a gaussien distribution N(0, sqrt(2/nl)), here nl is the total number of units in the 
    inputs tensor, nl = k²*c, here k is the spartial kernels size,  
    Args:
        ksize: filter size
        num_channels: the number of channels in the filter tensor
    Returns:
        the initalized weights
    """
    std = math.sqrt(2. / ksize ** 2 * num_channels)
    return tf.truncated_normal_initializer(stddev=std)
    