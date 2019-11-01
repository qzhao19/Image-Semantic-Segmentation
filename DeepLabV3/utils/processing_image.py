import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def read_image_label(image_filename, label_filename, shape=(256, 256, 3)):
    """This function is to read rbg image and the ground truth image.
    Args:
        image_filename: string, image filename
        label_filename: string, label image filename
        shape: int tuple, image and label shape
    Returns:
        image object with default uint8 type [0, 255]
    """
    # read image and label using opencv, so we get [blue, green, red] channel
    bgr_image = cv2.imread(image_filename)
    bgr_label = cv2.imread(label_filename)

    # resize image
    bgr_image = cv2.resize(bgr_image, (shape[0], shape[1]), interpolation=cv2.INTER_NEAREST)
    bgr_label = cv2.resize(bgr_label, (shape[0], shape[1]), interpolation=cv2.INTER_NEAREST)

    if (len(bgr_image.shape) != 3) | (len(bgr_label.shape) != 3):
        print('Warning: grey image!', image_path.split('/')[-1])
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
        bgr_label = cv2.cvtColor(bgr_label, cv2.COLOR_GRAY2BGR)
        
    # convert bgr image/label into a rgb image/label
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)

    return np.asarray(rgb_image), np.asarray(rgb_label)
    

def normalize_image(images, mode='tf'):
    """image preprocessing, Zero-center by mean pixel, it will scale pixels between -1 and 1.
    Args:
        images: 4D image or tensor with [batch_size, height, width, channel]
        mode: One of "tf", "torch" or "caffe".
            - caffe:then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.
            - torch: will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset.
    """
    if images.get_shape().ndims != 3:
        raise ValueError('Input must have size [height, width, channel>0]')
    
    if mode == 'tf':
        images /= 127.5
        images -= 1.0

    elif mode == 'torch':
        images /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_channels = images.get_shape().as_list()[-1]
        channels = tf.split(images, num_or_size_splits=num_channels, axis=2)
        for i in range(num_channels):
            channels[i] -= mean[i]
            channels[i] /= std[i]
        images = tf.concat(channels, axis=2)

    elif mode == 'caffe':
        mean = [123.68, 116.779, 103.939]
        num_channels = images.get_shape().as_list()[-1]
        channels = tf.split(images, num_or_size_splits=num_channels, axis=2)
        for i in range(num_channels):
            channels[i] -= mean[i]
        images = tf.concat(channels, axis=2)
    else:
        raise ValueError('Preprocessing image mode should be one of 3 method (tf, torch, caffe)')

    return images



def random_flip_image_and_label(image, label):
    """Randomly flip an image and label horizontally (left to right).
    Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`
    Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
    """
    image = tf.image.random_flip_left_right(image)
    label = tf.image.random_flip_left_right(label)

    return image, label


