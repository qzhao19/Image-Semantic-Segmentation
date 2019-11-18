import tensorflow as tf
from processing_image import preprocess_image


def image_label_batch_generator(image_label_set, shape, batch_size, buffer_size, mode='tf', is_shuffle=True):
    """Read tfrecords file using tensorflow TFRecordDataset module, which means a tfrecord data generator
    Args:
        image_label_set: ndarray, tuple, images and labels set
        shape: 3D tensor shape of [height, width, channel]
        batch_size: int, batch size 
        num_images: int, the number of example in the dataset, for perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required
        mode: string, imahe normalization methods, default values are 'tf', 'torch' and 'caffe'
        is_shuffle: boolean, shuffle dataset
    Returns:
        A tuple of images and labels one batch by one batch

    """
    # read image label dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_label_set)

    if is_shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(lambda image, label: preprocess_image(image, label, shape, mode))
    dataset = dataset.prefetch(batch_size) # software pipelining

    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = tf.data.make_one_shot_iterator(dataset)
    image, label = iterator.get_next()

    return image, label
