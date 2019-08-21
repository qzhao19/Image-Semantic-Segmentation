import os
import tensorflow as tf
from model import segnet
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

flags.DEFINE_string('prefix', '', 'The prefix is the affix which is placed before the image or label path')

flags.DEFINE_integer('num_classes', 13, 'The number of label classes')

flags.DEFINE_integer('input_height', 64, 'The image height')

flags.DEFINE_integer('input_width', 64, 'The image width')

flags.DEFINE_integer('input_channel', 3, 'The image channels')

flags.DEFINE_float('base_learning_rate', 0.0001, "base learning rate for optimizer")

FLAGS = flags.FLAGS



def train(FLAGS):
    """training model
    
    
    """
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    
    valid_steps = FLAGS.valid_steps
    max_steps = FLAGS.max_steps
    
    batch_size = FLAGS.batch_size
    train_file = FLAGS.train_file
    valid_file = FLAGS.valid_file
    num_classes = FLAGS.num_classes
    
    tb_logs = FLAGS.tb_logs
    
    height, width, channel = FLAGS.input_height, FLAGS.input_width, FLAGS.input_channel
    
    
    train_image_filename, train_label_filename = get_filename_list(FLAGS, 'train')
    valid_image_filename, valid_label_filename = get_filename_list(FLAGS, 'valid')
    
    with tf.Graph().as_default():
        
        # define tensor placeholder
        images_pl = tf.placeholder(tf.float32, shape=[batch_size, height, width, channel])
        labels_pl = tf.placeholder(tf.int32, shape=[batch_size, height, width, 1])
        is_training_pl = tf.placeholder(tf.bool, name='is_training')
        
        # define variable global_steps 
        global_steps = tf.Variable(0, trainable=False)
        
        # read image as tensor
        train_images, train_labels = dataset_inputs(train_image_filename, train_label_filename, batch_size, FLAGS)
        valid_images, valid_labels = dataset_inputs(valid_image_filename, valid_label_filename, batch_size, FLAGS)
        
        # build a graph that compute the logits prediction from model
        logits = segnet(images_pl, labels_pl, num_classes, is_training_pl)
        
        loss, accuracy, _ = calc_loss(logits, labels_pl, num_classes)
        
        # build a graph that trains the model with one batch of example and updates the model params 
        training_op = train_op(loss, global_steps, FLAGS.base_learning_rate)
        
        # define the model saver
        saver = tf.train.Saver(tf.global_variables())
        
        # define a summary operation 
        summary_op = tf.summary.merge_all()
        
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
    
            # start the queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_writter = tf.summary.FileWriter(tb_logs, sess.graph)
            
            # start to training
            for step in range(max_steps):
                # transform images and labels as 4D tensor 
                train_image_batch, train_label_batch = sess.run([train_images, train_labels])
                
                train_feed_dict = {images_pl: train_image_batch, 
                                   labels_pl: train_label_batch, 
                                   is_training_pl: True} 
               
                _, _loss, _acc, _summary_op = sess.run([training_op, loss, accuracy, summary_op], feed_dict=train_feed_dict)
                
                train_loss.append(_loss)
                train_acc.append(_acc)
                print('Iteration {}: train loss{:6.3f}, train accuracy{:6.3f}'.format(step, train_loss[-1], train_acc[-1]))
                
                if step % 10 == 0:
                    _logits = sess.run(logits, feed_dict=train_feed_dict)
                    print('Per class accuracy by logits in training time', per_class_acc(_logits, train_label_batch))
                    train_writter.add_summary(_summary_op, step)
                    
                if step % 100 == 0:
                    print('start validation process')
                    _valid_loss, _valid_acc = [], []
                    hist = np.zeros((num_classes, num_classes))
                    for step in range(int(valid_steps)):
                        valid_images_batch, valid_labels_batch = sess.run([valid_images, valid_labels])
                        
                        valid_feed_dict = {images_pl: valid_images_batch, 
                                           labels_pl: valid_labels_batch, 
                                           is_training_pl: True}
                        
                        _loss, _acc, _logits = sess.run([loss, accuracy, logits], feed_dict=valid_feed_dict)
                        
                        _valid_loss.append(_loss)
                        _valid_acc.append(_acc)
                        
                        hist += get_hist(_logits, valid_labels_batch)
                        
                    print_hist_summary(hist)
                    
                    # compute mean loss and acc mean
                    valid_loss.append(np.mean(_valid_loss))
                    valid_acc.append(np.mean(_valid_acc))
                   
                    print("Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                          step, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1]))
                    
                np.save(os.path.join(FLAGS.save_dir, "train_loss"), train_loss)
                np.save(os.path.join(FLAGS.save_dir, "train_acc"), train_acc)
                np.save(os.path.join(FLAGS.save_dir, "valid_loss"), valid_loss)
                np.save(os.path.join(FLAGS.save_dir, "valid_acc"), valid_acc)
                checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
            coord.request_stop()
            coord.join(threads)
            
        
def main(args):
    train(FLAGS)

        
if __name__ == '__main__':
    tf.app.run()       
    