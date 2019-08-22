import os
import tensorflow as tf
from model import segnet
from draws import save_images
from inputs import get_test_data
from eval import calc_loss, train_op, per_class_acc, get_hist, print_hist_summary


flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_integer('valid_steps', 21, 'The number of validation steps ')

flags.DEFINE_integer('max_steps', 301, 'The number of maximum steps for traing')

flags.DEFINE_integer('batch_size', 10, 'The number of images in each batch during training')

flags.DEFINE_string('train_file', './carla_dataset/train.txt', 'The path to training file names')

flags.DEFINE_string('valid_file', './carla_dataset/valid.txt', 'The path to validation file names')

flags.DEFINE_string('test_file', './carla_dataset/test.txt', 'The path to test file names')

flags.DEFINE_string('tb_logs', './tensorboard_logs/', 'The path points to tensorboard logs ')

flags.DEFINE_string('save_dir', './segnet_model', 'The path to saved checkpoints')

flags.DEFINE_string('ckpt_dir', '', """ checkpoint file """)

flags.DEFINE_string('prefix', '', 'The prefix is the affix which is placed before the image or label path')

flags.DEFINE_integer('num_classes', 13, 'The number of label classes')

flags.DEFINE_integer('input_height', 64, 'The image height')

flags.DEFINE_integer('input_width', 64, 'The image width')

flags.DEFINE_integer('input_channel', 3, 'The image channels')

flags.DEFINE_float('base_learning_rate', 0.0001, "base learning rate for optimizer")

FLAGS = flags.FLAGS


def inference(FLAGS):
    """test model
    """
    
    valid_steps = FLAGS.valid_steps
    max_steps = FLAGS.max_steps
    
    batch_size = FLAGS.batch_size
    train_file = FLAGS.train_file
    valid_file = FLAGS.valid_file
    num_classes = FLAGS.num_classes
    
    tb_logs = FLAGS.tb_logs
    
    ckpt_dir = FLAGS.ckpt_dir
    
    height, width, channel = FLAGS.input_height, FLAGS.input_width, FLAGS.input_channel
    
    test_image_filename, test_label_filename = get_filename_list(FLAGS, 'test')
    
    with tf.Graph().as_default():
        
        
        # define tensor placeholder
        images_pl = tf.placeholder(tf.float32, shape=[batch_size, height, width, channel])
        labels_pl = tf.placeholder(tf.int32, shape=[batch_size, height, width, 1])
        is_training_pl = tf.placeholder(tf.bool, name='is_training')
                
        
        # build a graph that compute the logits prediction from model
        logits = segnet(images_pl, labels_pl, num_classes, is_training_pl)
        
        
        loss, acc, predicts = calc_loss(logits, labels_pl, num_classes)
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_dir)
            
            hist = np.zeros((num_classes, num_classes))
            
            images, labels = get_test_data(test_image_filename, test_label_filename)
        
            threads = tf.train.start_queue_runners(sess=sess)
            
            for images_batch, labels_batch  in zip(images, labels):
                feed_dict = {images_pl: images_batch,
                             labels_pl: labels_batch,
                             is_training_pl: False }
                
                _logits, _predicts = sess.run([logits, predicts], feed_dict=feed_dict)
                
                if (FLAGS.save_image):
                    save_images(_predicts[0], 'testing_image.png')
                    
                hist += get_hist(dense_prediction, label_batch)
    
            print_hist_summary(hist)
    