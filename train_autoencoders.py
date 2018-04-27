import tensorflow as tf
import pandas as pd
from autoencoder import StackedAutoencoderClassifier
import data_helpers
import os
import time 
import numpy as np 
import datetime 
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name='ssda'

def train(train_x, train_y, test_x, test_y, feature_num):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            sa = StackedAutoencoderClassifier(
                    n_input=feature_num,
                    n_hidden_list=[300, 400, 500],
                    num_classes=len(train_y[0]),
                    l2_reg_lambda=0.1
                    )
    
            timestamp = str(int(time.time()))
            out_dir = '../runs/'+ model_name + '-' + timestamp
            print('Writing to {}\n'.format(out_dir))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    
            sess.run(tf.global_variables_initializer())
    
            def test(batches):
                print('Evaluate')
                all_predictions = []
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    x_batch, y_batch = list(x_batch), list(y_batch)
                    batch_predictions = sess.run(sa.predictions, feed_dict={sa.input_x: x_batch})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

                y_test = np.array(list(map(lambda a: a.index(1), test_y)))
                correct_predictions = sum(all_predictions == y_test)
                print('Total number of test examples: {}'.format(len(y_test)))
                print("Accuracy: {:g}".format(correct_predictions/len(y_test)))
                return correct_predictions / len(y_test)

            # pre training
            batches = data_helpers.batch_iter_numeric(train_x, train_y, 256, 30)
            for i in range(len(sa.autoencoders)):
                for batch in batches:
                    x_batch, _ = zip(*batch)
                    x_batch = list(x_batch)
                    for j in range(i):
                        x_batch = sa.autoencoders[j].comput_hidden(x_batch, sa.autoencoders[j].hidden_weight, sa.autoencoders[j].hidden_bias, False)
                    if i > 0:
                        x_batch = sess.run(x_batch)
                    _, loss = sess.run([sa.autoencoders[i].train_op, sa.autoencoders[i].loss], feed_dict={sa.autoencoders[i].input_x: x_batch})
                    print('{} Layer: loss {:g}'.format(i, loss))
    
            #f = open('out.txt', 'w')
            #print('pre train value of W, b')
            #for encoder in sa.autoencoders:
            #    print(sess.run(encoder.hidden_weight), sess.run(encoder.hidden_bias), file=f)
    
            #f.close()
            # fine tune
            # global_step = tf.Variable(0, name='global_step', trainable=False)
            # sess.run(global_step.initializer)
            print('\nfine tune')
            batches = data_helpers.batch_iter_numeric(train_x, train_y, 256, 50)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, y_batch = list(x_batch), list(y_batch)
                feed_dict = {
                        sa.input_x: x_batch,
                        sa.input_y: y_batch 
                        }
                _, loss, accuracy = sess.run([sa.train_op, sa.loss, sa.accuracy], feed_dict)
                            
                time_str = datetime.datetime.now().isoformat()
                print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                # current_step = tf.train.global_step(sess, global_step)
                # if current_step % 100 == 0:
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))
            
            #f = open('out2.txt', 'w')
            #print('after fine tune value of W, b')
            #for encoder in sa.autoencoders:
            #    print(sess.run(encoder.hidden_weight), sess.run(encoder.hidden_bias), file=f)
            #f.close()
            
            # evaluate
            batches = data_helpers.batch_iter_numeric(test_x, test_y, 128, 1, False) 
            all_predictions = []
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, y_batch = list(x_batch), list(y_batch)
                batch_predictions = sess.run(sa.predictions, feed_dict={sa.input_x: x_batch})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            y_test = np.array(list(map(lambda a: a.index(1), test_y)))
            data_helpers.save_confusion_matrix(y_test, all_predictions, '../confusion-matrix/'+model_name + '-' + timestamp)
            correct_predictions = sum(all_predictions == y_test)
            print('Total number of test examples: {}'.format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/len(y_test)))
            return correct_predictions / len(y_test)

accuracy = 0.0
l = int(sys.argv[1])
for i in range(l):
    x_train, y_train = data_helpers.load_numeric_feature_and_labels('../data/trainingset' + str(i+1) + '.txt')
    x_test, y_test = data_helpers.load_numeric_feature_and_labels('../data/testset' + str(i+1) + '.txt')
    feature_num = len(x_train[0])
    accuracy += train(x_train, y_train, x_test, y_test, feature_num)

print('Average accuracy: {:g}'.format(accuracy/l))

