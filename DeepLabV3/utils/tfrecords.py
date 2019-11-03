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
  

def _parse_tfrecords(serialized_example):
    """Function focus on read tfrecords file
    Args:
        serialized_example: 
    """

    features = {'height': tf.FixedLenFeature([], tf.int32), 
                'width': tf.FixedLenFeature([], tf.int32), 
                'channel': tf.FixedLenFeature([], tf.int32),
                'image': tf.FixedLenFeature([], tf.string),
                'image_format': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'label_format': tf.FixedLenFeature([], tf.string)}

    # get features from serialized tf example
    parsed_features = tf.parse_single_example(serialized_example, features=features)

    # get image height, width and channel value
    height = parsed_features['height']
    width = parsed_features['width']
    channel = parsed_features['channel']

    # get image data, set image type up float32
    image = tf.io.decode_png(parsed_features['image'], channels=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape([None, None, 3])

    # get label data, set label type up int32
    label = tf.io.decode_png(parsed_features['label'], channels=1)
    label = tf.image.convert_image_dtype(label, dtype=tf.int32)
    label.set_shape([None, None, 1])

    return image, label


def read_tfrecords(tfrecords_file, shape, batch_size, num_images, num_epochs, mode='tf', is_training=True, is_shuffle=True):
    """Read tfrecords file using tensorflow TFRecordDataset module, which means a tfrecord data generator
    Args:
        tfrecords_file: string, tfrecords file path
        buffer_size: int, the number of example in the dataset, for perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required
        is_shuffle: boolean, shuffle dataset
    Returns:
        A tuple of images and labels one batch by one batch

    """
    # read tfrecord file
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    if shuffle:
        dataset = dataset.shffle(num_images)

    dataset = dataset.map(_parse_tfrecords)
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, shape, mode, is_training))
    dataset = dataset.prefetch(batch_size) # software pipelining

    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()

    return image, label

  
