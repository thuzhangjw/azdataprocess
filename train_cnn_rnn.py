import tensorflow as tf
import pandas as pd
from gensim.models import Word2Vec 
import time
import os
import datetime
import data_helpers 
from cnn_rnn import CNN_RNN
import numpy as np
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(x_train, y_train, x_test, y_test):
    max_sentence_num = max([len(a) for a in (x_train + x_test)])
    max_sentence_length = 0
    for doc in (x_train + x_test):
        for sentence in doc:
            sl = len(sentence.strip().split(' '))
            if sl > max_sentence_length:
                max_sentence_length = sl
    
    init_words_embedded_model = Word2Vec.load('../data/word2vec.model')
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = CNN_RNN(
                    sentence_num=max_sentence_num,
                    sentence_length=max_sentence_length,
                    filter_sizes=[2,3,4,5],
                    num_filters=64,
                    init_words_embedded_model=init_words_embedded_model,
                    rnn_hidden_dim=128,
                    num_classes=len(y_train[0]),
                    l2_reg_lambda=0.2,
                    use_static=True
                    )

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(cnn_rnn.loss, global_step=global_step)
    
            timestamp = str(int(time.time()))
            out_dir = '../runs/cnn_rnn-' + timestamp 
            print("Writing to {}\n".format(out_dir))
    
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            batches = data_helpers.batch_iter_text_matrix(x_train, y_train, 128, 50, max_sentence_length, max_sentence_num)
            for batch in batches:
                x_num_batch, y_batch = zip(*batch)
                x_batch, real_sentence_num_batch = zip(*x_num_batch)
                x_batch, y_batch, real_sentence_num_batch = list(x_batch), list(y_batch), list(real_sentence_num_batch)
                
                feed_dict = {
                        cnn_rnn.input_x: x_batch,
                        cnn_rnn.input_y: y_batch,
                        cnn_rnn.dropout_keep_prob: 0.5,
                        cnn_rnn.batch_size: len(y_batch),
                        cnn_rnn.real_sentence_num: real_sentence_num_batch
                        }

                _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))            
            
            # evaluate
            print('Evaluate')
            all_predictions = []
            batches = data_helpers.batch_iter_text_matrix(x_test, y_test, 128, 1, max_sentence_length, max_sentence_num, False)
            for batch in batches:  
                x_num_batch, y_batch = zip(*batch)
                x_batch, real_sentence_num_batch = zip(*x_num_batch)
                x_batch, y_batch, real_sentence_num_batch = list(x_batch), list(y_batch), list(real_sentence_num_batch)
                
                feed_dict = {
                        cnn_rnn.input_x: x_batch,
                        cnn_rnn.dropout_keep_prob: 1.0,
                        cnn_rnn.batch_size: len(y_batch),
                        cnn_rnn.real_sentence_num: real_sentence_num_batch
                        }

                batch_predictions = sess.run(cnn_rnn.predictions, feed_dict)
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            y_test = np.array(list(map(lambda a: a.index(1), y_test)))
            correct_predictions = sum(all_predictions == y_test)
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/len(y_test)))
            return correct_predictions / len(y_test)


accuracy = 0.0
l = int(sys.argv[1])
for i in range(l):
    x_train, y_train = data_helpers.load_sentences_matrix__and_labels('../data/trainingset' + str(i+1) + '.txt')
    x_test, y_test = data_helpers.load_sentences_matrix__and_labels('../data/testset' + str(i+1) + '.txt')
    accuracy += train(x_train, y_train, x_test, y_test) 

print('Average accuracy: {:g}'.format(accuracy/l))
    


