import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def read_image_label(image_filename_list, label_filename_list):
    """This function is to read rbg image and the ground truth image.
    Args:
        image_filename_list: list of string, an image filename
        label_filename_list: list of string, a label image filename
    Returns:
        image object with default uint8 type [0, 255]
    """
    images = []
    labels = []

    for image_filename, label_filename in zip(image_filename_list, label_filename_list):
        if not os.path.exists(image_filename):
            raise ValueError('Error: %s not exists' %image_filename)
            continue
        if (image_filename.split('/')[-1].split('.')[-1] not in ['png']) and (label_filename.split('/')[-1].split('.')[-1] not in ['png']):
            raise ValueError('The format of image must be png')
            continue
        bgr_image = cv2.imread(image_filename)
        bgr_label = cv2.imread(label_filename)
        
        # convert bgr image/label into a rgb image/label
        # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)
        
        rgb_label = rgb_label[:, :, 0].reshape((rgb_label.shape[0], rgb_label.shape[1], 1))           
        images.append(rgb_image)
        labels.append(rgb_label)

    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.uint8)


def normalize_image(images, mode):
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

def resize_image_label(image, label, shape):
    """Resize image and label  
    Args:
        image: 3D tensor of shape [height, width, channel]
        label: 3D tensor of shape [height, width, 1]
        shape: tuple of image shape [height, width, channel]
    Returns:
        A 3-D tensor of the same type and shape as `image`.
        A 3-D tensor of the same type and shape as `label`.
    """

    image = tf.image.resize_images(image, (shape[0], shape[1]), method=tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize_images(label, (shape[0], shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label

def preprocess_image(image, label, shape, mode):
    """Image preprocessing interface, if is_training procedure, we want to normalize and randomly flip image, if not we just resize image 
    Args:
        image: 3D tensor 
        label: 3D tensor
        is_training: boolean, if is training
    """
    
    # resize image and label
    image, label = resize_image_label(image, label, shape)
        
    image = normalize_image(image, mode=mode)

    return image, label

