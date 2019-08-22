import tensorflow as tf
from scipy import misc

def get_filename_list(FLAGS, mode='train'):
    """get the filename list
    Args:
        FLAGS: implement command line flags 
        mode: learning type with 3 default value: train, valid, test
    Returns:
        image file name and label file name
    
    """
    if mode == 'train':
        path = FLAGS.train_file
    elif mode == 'valid':
        path = FLAGS.valid_file
    elif mode == 'test':
        path = FLAGS.test_file
    else:
        raise ValueError('Please check the learning mode')
    
    image_filenames = []
    label_filenames = []
    with open(path) as file:
        lines = file.readlines()
        lines = [line.strip().split(' ') for line in lines] 

    for i in range(len(lines)):
        image_filenames.append(lines[i][0])
        label_filenames.append(lines[i][1])

    image_filenames = [FLAGS.prefix + filename for filename in image_filenames]
    label_filenames = [FLAGS.prefix + filename for filename in label_filenames]
    
    return image_filenames, label_filenames


def dataset_reader(filename_queue, FLAGS):  # prev name: CamVid_reader
    """read image and label as tensor
    Args:
        filename_queue: a string queue
        config: config file
    Returns:
        return image and label tensor
    
    """
    image_filenames = filename_queue[0]  # tensor of type string
    label_filenames = filename_queue[1]  # tensor of type string

    # get png encoded image
    image_value = tf.read_file(image_filenames)
    label_value = tf.read_file(label_filenames)
    
    
    # decodes a png image into a uint8 or uint16 tensor
    # returns a tensor of type dtype with shape [height, width, depth]
    image_bytes = tf.image.decode_png(image_value)
    label_bytes = tf.image.decode_png(label_value, 1)  # Labels are png, not jpeg

    image_resized = tf.image.resize_images(image_bytes, (FLAGS.input_height, FLAGS.input_width), method=tf.image.ResizeMethod.BILINEAR)
    label_resized = tf.image.resize_images(label_bytes, (FLAGS.input_height, FLAGS.input_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    
    image = tf.cast(tf.reshape(image_resized, (FLAGS.input_height, FLAGS.input_width, FLAGS.input_channel)), tf.float32)
    label = tf.cast(tf.reshape(label_resized, (FLAGS.input_height, FLAGS.input_width, 1)), tf.int64)
    
    
    return tf.image.per_image_standardization(image), label


def dataset_inputs(image_filenames, label_filenames, batch_size, FLAGS):
    """
    
    """
    image_tensor = tf.convert_to_tensor(image_filenames, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(label_filenames, dtype=tf.string)
    
    
    filename_queue = tf.train.slice_input_producer([image_tensor, label_tensor], shuffle=True)

    images, labels = dataset_reader(filename_queue, FLAGS)
        
    # reshaped_image = tf.cast(image, tf.float32)
    # reshaped_label = tf.cast(label, tf.int64)
    
    min_queue_examples = 100
    
    print('Filling queue with %d input images before starting to train. This may take some time.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_label_batch(images, labels, min_queue_examples, batch_size)


def generate_image_label_batch(image, label, min_queue_examples, batch_size, shuffle=True):
    """Construct a queued batch of images and labels.
    Args:
        image: 3D Tensor of [height, width, 3] of type.float32.
        label: 3D Tensor of [height, width, 1] type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.

    num_preprocess_threads = 1
    if shuffle:
        images_batch, label_batch = tf.train.shuffle_batch([image, label], 
                                                           batch_size=batch_size,
                                                           num_threads=num_preprocess_threads,
                                                           capacity=min_queue_examples + 3 * batch_size,
                                                           min_after_dequeue=min_queue_examples)
    else:
        images_batch, label_batch = tf.train.batch([image, label],
                                                   batch_size=batch_size,
                                                   num_threads=num_preprocess_threads,
                                                   capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('training_images', images_batch)
    print('generating image and label batch:')
    return images_batch, label_batch


def get_test_data(image_filenames, label_filenames, FLAGS):
    images = []
    labels = []
    index = 0
    for image_filename, label_filename in zip(image_filenames, label_filenames):
        image = scipy.misc.imread(image_filename)
        label = scipy.misc.imread(label_filename)
        image = misc.imresize(image, (FLAGS.input_height, FLAGS.input_width))
        label = misc.imresize(label, (FLAGS.input_height, FLAGS.input_width))
        images.append(image)
        labels.append(label)
        index = index + 1

    print('%d CamVid test images are loaded' % index)
    return images, labels
