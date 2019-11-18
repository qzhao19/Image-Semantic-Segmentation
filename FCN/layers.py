import tensorflow as tf
import math


def max_pooling(inputs, ksize, strides, padding, name):
    """MaxPooling layer for ConvNet
    Args:
        inputs: 4D input tensor with [batchs_size, height, width, channel]
        ksize: int list variable kernel size with [kernel_height, kernel_width]
        stride: int list varible stride with [stride left, stride right]
        padding: string padding mode
        name: maxpooling name for each layer
    Returns:
        A 4D tensor 
    """
    # define name scope
    with tf.variable_scope(name) as scope:
        value = tf.nn.max_pool(inputs, ksize=[1, ksize[0], ksize[1], 1], strides=[1, strides[0], strides[1], 1], padding='SAME', name=scope.name)

    return tf.cast(value, tf.float32)


def up_sampling(pool, name):
    """Unpooling layer after max_pool
    
    Args:
       pool:   max pooled output tensor
    
    """
    with tf.variable_scope(name):
        pool_shape = tf.shape(pool)
        return tf.image.resize_images(pool, (2*pool_shape[1], 2*pool_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def batch_norm(inputs, is_training, scope):
    """Batch normalization allows each layer of a network to learn by itself a little bit more independantly of other layer. batch normalization normalizes the output of previous 
    layer by substracting mean value of a batch data, and dividing by the batch standard deviation. In the orginaml paper , they used local reponse normalization method to normalize 
    the outputs of each layer of a network
    Args:
        inputs: 4D tensor with float32
        is_training: bool variable, for training dataset, is_training should be true, for test dataset, it must be false
        name: layer name
    Returns:
        4D tensor
    """
    # create a scope name
    with tf.variable_scope(scope.name) as scope:
        return tf.cond(is_training,
                       lambda: tf.contrib.layers.batch_norm(inputs, center=True, is_training=True, scope=scope), 
                       lambda: tf.contrib.layers.batch_norm(inputs, center=False, is_training=False, reuse=True, scope=scope))

    
def variable_with_weights_decay(shape, initializer, wd, name):
    """This function aims to creating initializer with weights decay, the variabel is initalized by the method he_normal with tf.initalizers.he_normal
    Args:
        shape: int list, presents inputs tensor shape
        initializer: weights initializer function, he using he method
        wd: weights decay coef
        name: layer name 
    Returns:
        weights intialized by he normal method
    """

    var = tf.get_variable(name, shape=shape, initializer=initializer)
    if wd:
        # calculate weights decay using l2 loss regularization technique, the aims is to overcome overfitting issue
        # add weights_decay into graph variable collection
        weights_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weights_loss')
        tf.add_to_collection('losses', weights_decay)
    return var


def conv2d(inputs, shape, strides, padding, is_training, name):
    """convolutional layer with weights decay
    Args:
        inputs: input tensor with shape [batch_size, height, width, channel]
        shape: int tuple, the kernel shape with [heights, width, input_depth, output_depth]
        stride: int tuple, the windows strides with 1x1 pixel
        padding: string, padding mode
        name: layer name
    Returns:
    """
    with tf.variable_scope(name) as scope:
        # first, using weight_decay function to intialize weight
        weights = variable_with_weights_decay(shape=shape, initializer=tf.initializers.he_normal(), wd=5e-4, name='weights')
        # weights = tf.get_variable('weights', shape=shape, initializer=tf.initializers.he_normal())
        # we intialize bias by using random normal without reguarization term
        # biases = tf.get_variable('biases', shape=[shape[3]], initializer=tf.initializers.zeros())
        biases = variable_with_weights_decay(shape=shape[3], initializer=tf.constant_initializer(0.0), wd=False, name='biases')
        outputs = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding)
        outputs = tf.nn.bias_add(outputs, bias=biases)
        outputs = tf.nn.relu(batch_norm(outputs, is_training, scope), name=scope.name+'batch_norm/relu')

    return outputs


def fc(inputs, shape, name):
    """Fully connected layer, fc layer takes outputs of convolution/pooling layer and predicts the best probabilities of label to describe image.
    It takes outputs of the previous layers, flatten them => trun them into a single vector. For vgg16 network, there are 3 fully connected layers
    Args:
        inputs: float32 input tensor 
        shape: int tuple, [heights, width, input_depth, output_depth]
        name: layer name
    Returns:
    """
    # define namespace
    with tf.variable_scope(name) as scope:
        # weights = tf.get_variable('weights', shape=shape, initializer=tf.initializers.he_normal())
        # first 2 fully connected layers are regularised by weights decay
        weights = variable_with_weights_decay(shape=shape, initializer=tf.initializers.he_normal(), wd=5e-4, name='weights')
        # we initialize bias using a constant 0.1 to avoid backdrop issue
        # biases = tf.get_variable('biases', shape=[shape[1]], initializer=tf.constant_initializer(tf.constant(0.1, dtype=tf.float32, shape=[shape[1]])))

        # biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[shape[1]]), name='biases')

        biases = tf.get_variable('biases', shape=[shape[1]], initializer=tf.initializers.random_normal())
        outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        outputs = tf.nn.relu(outputs, name=scope.name)
    return outputs


def dropout(inputs, keep_prob, name):
    """dropout layer, we use dropout regularisation for 2 first fully connected layers
    Args:
        inputs: 4D input tensor with float32
        keep_prob: the probability of keep training sample
        name:
    Returns:
        
    """
    return tf.nn.dropout(inputs, rate=(1-keep_prob), name=name)
