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
    
    
