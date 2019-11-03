import os
import tensorflow as tf
from utils import load_image_label_file
from processing_image import read_image_label


def int64_feature(value):
    """create int32 type feature 
    """
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    """create bytes type feature 
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(file_path, tfrecords_file):
    """Function takes label and image height, width, channel etc, to store as tfrecord data 
    Args:
        file_path: the path to image_label_file
        shape: int tuple, containing image height and image width
        output_tfrecords_dir: the path points output_tfrecords_dir
    Raises:
        ValueError: 
    """
    # get images and labels list, and make sure the number of images and labels is equal
    image_filename_list, label_filename_list = load_image_label_file(file_path)
    if len(image_filename_list) != len(label_filename_list):
        raise ValueError('The size of image dose not match with that of label.')
    
    # create a tfrecord writer
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecords_file)
    for i, [image_filename, label_filename] in enumerate(zip(image_filename_list, label_filename_list)):
        if not os.path.exists(image_filename):
            raise ValueError('Error: image %s not exists' %image_filename)
            continue

        if not os.path.exists(label_filename):
            raise ValueError('Error: label %s not exists' %label_filename)
            continue

        # check image format
        if image_filename.split('/')[-1].split('.')[-1] not in ['png']:
            raise ValueError('The format of image must be png')
            continue

        # make sure label format
        if label_filename.split('/')[-1].split('.')[-1] not in ['png']:
            raise ValueError('The format of image must be png')
            continue

        image, label = read_image_label(image_filename, label_filename)
        image_bytes = image.tostring()
        label_bytes = label.tostring()
        if i%100 == 0 or i == (len(image_filename)-1):
            print('Current image_path=%s' %(image_filename), 'shape:{}'.format(image.shape), \
                  'Current label_path=%s' %(label_filename), 'shape:{}'.format(label.shape))
        
        
        tf_example = tf.train.Example(features=tf.train.Features(feature={'height': int64_feature(image.shape[0]), 
                                                                          'width': int64_feature(image.shape[1]), 
                                                                          'channel': int64_feature(image.shape[2]),
                                                                          'image': bytes_feature(image_bytes),
                                                                          'image_format': bytes_feature('png'.encode('utf8')),
                                                                          'label': bytes_feature(label_bytes),
                                                                          'label_format': bytes_feature('png'.encode('utf8'))}))
        try:
            tfrecord_writer.write(tf_example.SerializeToString())
        except ValueError:
            print('Invalid example, ignoring')      
    tfrecord_writer.close()
    print('TF record file has been done')
  
  
