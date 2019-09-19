import tensorflow as tf
import numpy as np



def normal_loss(logits, labels, num_classes):
    """This function focus on computing normal loss.
    
    Args:
        logits: 4D tensor. output tensor from segnet model, which is the output of the decode without softmax
        labels: true label tensor
        num_classes: the number of classes for the dataset 
    Returns:
        loss, accuracy, prediction(logits with softmax) 
    
    """
    # flatten the labels
    labels_flatten = tf.reshape(labels, [-1])
    labels_one_hot = tf.one_hot(labels_flatten, depth=num_classes)
    logits_reshape = tf.reshape(logits, [-1, num_classes])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_one_hot, 
                                                                   logits=logits_reshape, 
                                                                   name='cross_entropy')
    
    # compute loss, which is cross entropy mean
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar(name='loss', tensor=cross_entropy_mean)
    
    # compute prediction for labels
    predicts = tf.argmax(logits_reshape, axis=-1)
    true_predicts = tf.equal(predicts, labels_flatten)
    
    # compute accuracy
    accuracy = tf.reduce_mean(tf.cast(true_predicts, tf.float32))
    tf.summary.scalar(name='loss', tensor=accuracy)
    
    return cross_entropy_mean, accuracy, predicts


def weighted_loss(logits, labels, num_classes, frequency):
    """This function focus on computing weighted loss. Here frequency represents balancing frequency for each label, the formul
    is following: frequency=ln(total_sample/sample(c)), total_sample is the toatl sample of pixels in images, sample(c) is the 
    number of pixels of class c in the images
    
    Args:
        logits: 4D tensor. output tensor from segnet model, which is the output of the decode without softmax
        labels: true label tensor
        num_classes: the number of classes for the dataset 
        frequency: the weights for each classes
    Returns:
        loss, accuracy, prediction(logits with softmax) 
    
    """
    # flatten the labels
    labels_flatten = tf.reshape(labels, [-1])
    labels_one_hot = tf.one_hot(labels_flatten, depth=num_classes)
    logits_reshape = tf.reshape(logits, [-1, num_classes])
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=labels_one_hot, 
                                                             logits=logits_reshape, 
                                                             pos_weight=frequency)
    # compute loss, which is cross entropy mean
    
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar(name='loss', tensor=cross_entropy_mean)
    
    # compute prediction for labels
    predicts = tf.argmax(logits_reshape, axis=-1)
    true_predicts = tf.equal(predicts, labels_flatten)
    
    # compute accuracy
    accuracy = tf.reduce_mean(tf.cast(true_predicts, tf.float32))
    tf.summary.scalar(name='loss', tensor=accuracy)
    
    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, axis=-1)
    
    
def calc_loss(logits, labels, num_classes):
    """
    
    """
    loss_weight = np.array([0.2595, 2.3614, 2.5640, 0.1417, 0.6823, 0.9051, 
                            0.3826, 1.8418, 2.6446, 0.2478, 0.1826, 1.0974, 0.2253])
    
    # class 0 to 12, but the class 11 is ignored, so maybe the class 11 is background!

    labels = tf.cast(labels, dtype=tf.int64)
    loss, accuracy, prediction = weighted_loss(logits, labels, num_classes=num_classes, frequency=loss_weight)
    return loss, accuracy, prediction
    
    

def train_op(total_loss, global_steps, base_learning_rate):
    """This function defines train optimizer 
    Args:
        total_loss: the loss value
        global_steps: global steps is used to track how many batch had been passed. In the training process, the initial value for global_steps = 0, here  
        global_steps=tf.Variable(0, trainable=False). then after one batch of images passed, the loss is passed into the optimizer to update the weight, then the global 
        step increased by one.
    Returns:
        the train optimizer
    """
    # get updated opration 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # base_learning_rate = 0.1
        # define learning rate decay strategy, here we used exponentiel_decay
        learning_rate_decay = tf.train.exponential_decay(base_learning_rate, global_steps, 1000, 0.0005)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_decay)
        print("Running with Adam Optimizer with learning rate:", learning_rate_decay)
        
        grads = optimizer.compute_gradients(total_loss)
        training_op = optimizer.apply_gradients(grads, global_step=global_steps)
        
    return training_op
       
    

def per_class_acc(predicts, labels):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    #labels = labels

    batch_size = predicts.shape[0]
    num_classes = predicts.shape[3]
    hist = np.zeros((num_classes, num_classes))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predicts[i].argmax(2).flatten(), num_classes)
    total_acc = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' %np.nanmean(total_acc))
    
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IoU  = %f' % np.nanmean(iou))
    for i in range(num_classes):
        if float(hist.sum(1)[i]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[i] / float(hist.sum(1)[i])
        print("class %d accuracy = %f " % (i, acc))


def fast_hist(a, b, n):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_hist(predicts, labels):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    num_classes = predicts.shape[3] 
    batch_size = predicts.shape[0]
    hist = np.zeros((num_classes, num_classes))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predicts[i].argmax(2).flatten(), num_classes)
    return hist



def print_hist_summary(hist):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    total_acc = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(total_acc))
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IoU  = %f' % np.nanmean(iou))
    for i in range(hist.shape[0]):
        if float(hist.sum(1)[i]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[i] / float(hist.sum(1)[i])
        print("class %d accuracy = %f " %(i, acc))
