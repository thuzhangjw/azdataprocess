import tensorflow as tf
import pandas as pd
from numeric_cnn import NumericCNN
import data_helpers 
import time
import os
import datetime 
import sys 
import numpy as np 


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(train_x, train_y, test_x, test_y, feature_num, feature_dim):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            ncnn = NumericCNN(
                    feature_num=feature_num,
                    feature_dim=feature_dim,
                    filter_sizes=[3, 4, 5, 6],
                    num_filters=32,
                    num_classes=len(y_train[0]),
                    l2_reg_lambda=0.2
                    )
            
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(ncnn.loss, global_step=global_step)
    
            timestamp = str(int(time.time()))
            out_dir = '../runs/ncnn-' + timestamp 
            print("Writing to {}\n".format(out_dir))
    
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    
            sess.run(tf.global_variables_initializer())
    
            batches = data_helpers.batch_iter_numeric(train_x, train_y, 128, 200)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, y_batch = list(x_batch), list(y_batch)
                feed_dict = {
                        ncnn.input_x: x_batch,
                        ncnn.input_y: y_batch,
                        ncnn.dropout_keep_prob: 0.5
                        }
                _, step, loss, accuracy = sess.run([train_op, global_step, ncnn.loss, ncnn.accuracy], feed_dict)
    
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))
        
            # evaluate
            all_predictions = []
            batches = data_helpers.batch_iter_numeric(test_x, test_y, 128, 1, False)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, y_batch = list(x_batch), list(y_batch)
                batch_predictions = sess.run(ncnn.predictions, feed_dict={ncnn.input_x: x_batch, ncnn.dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            y_test = np.array(list(map(lambda a: a.index(1), test_y)))
            correct_predictions = sum(all_predictions == y_test)
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/len(y_test)))
            return correct_predictions / len(y_test)

accuracy = 0.0
l = int(sys.argv[1])
for i in range(l):
    x_train, y_train = data_helpers.load_numeric_matrix_data('../data/trainingset' + str(i+1) + '.txt')
    x_test, y_test = data_helpers.load_numeric_matrix_data('../data/testset' + str(i+1) + '.txt')
    feature_num = len(x_train[0])
    feature_dim = len(x_train[0][0])
    accuracy += train(x_train, y_train, x_test, y_test, feature_num, feature_dim)

print('Average accuracy: {:g}'.format(accuracy/l))

