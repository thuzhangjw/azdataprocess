import tensorflow as tf
from cnn import CNNLayer
import pandas as pd
from gensim.models import Word2Vec
import os
import data_helpers
import datetime
import time
from numeric_cnn import NumericCNN
from cnn_rnn import CNN_RNN
import sys
import numpy as np
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name='whole'

def add_noise(batch, ratio, feature_ratio, replace_prob, feature_dim_list):
    for data in batch:
        if random.random() <= ratio:
            start_index = 0
            end_index = 0
            for idx, feature in enumerate(data):
                end_index += feature_dim_list[idx]
                if random.random() <= feature_ratio:
                    if 1 in feature:
                        index_1 = feature.index(1)
                        feature[index_1] = 0.0
                    if random.random() <= replace_prob:
                        new_index = random.randint(start_index, end_index-1)
                        feature[new_index] = 1.0
                start_index = end_index

def train(x_train_cnn, x_train_cnn_rnn, x_train_ncnn, y_train, x_test_cnn, x_test_cnn_rnn, x_test_ncnn, y_test, numeric_feature_dim_list):

    max_accuracy = 0.0
    cor_step = 0
    cnn_max_sentence_length = max([len(x.split(' ')) for x in (x_train_cnn + x_test_cnn)])
    ncnn_feature_num = len(x_train_ncnn[0])
    ncnn_feature_dim = len(x_train_ncnn[0][0])
    
    cnn_rnn_max_sentence_num = max([len(a) for a in (x_train_cnn_rnn + x_test_cnn_rnn)])
    cnn_rnn_max_sentence_length = data_helpers.get_cnn_rnn_max_sentence_length(x_train_cnn_rnn, x_test_cnn_rnn)
    init_words_embedded_model = Word2Vec.load('../data/word2vec.model')
    num_classes = len(y_train[0])
 
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True 
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            input_y = tf.placeholder(tf.float32, [None, num_classes], name='final_input_y')
            dropout_keep_prob = tf.placeholder(tf.float32, [], name='whole_dropout_keep_prob')
            cnn = CNNLayer(
                    sequence_length=cnn_max_sentence_length,
                    filter_sizes=[2,3,4,5],
                    num_filters=100,
                    init_words_embedded_model=init_words_embedded_model,
                    num_classes=num_classes,
                    l2_reg_lambda=0.5,
                    use_static=True
                    )
            

            cnn_rnn = CNN_RNN(
                    sentence_num=cnn_rnn_max_sentence_num,
                    sentence_length=cnn_rnn_max_sentence_length,
                    filter_sizes=[2,3,4,5],
                    num_filters=100,
                    init_words_embedded_model=init_words_embedded_model,
                    rnn_hidden_dim=300,
                    num_classes=num_classes,
                    l2_reg_lambda=0.25,
                    use_static=True 
                    )
            ncnn = NumericCNN(
                    feature_num=ncnn_feature_num,
                    feature_dim=ncnn_feature_dim,
                    filter_sizes=[2,3,4,5,7,8,11,13,19,23,25],
                    num_filters=50,
                    num_classes=num_classes,
                    l2_reg_lambda=0.2
                    )
            
            merged_feature = tf.concat([cnn.final_feature, cnn_rnn.final_feature, ncnn.final_feature], 1)
            #merged_feature = ncnn.final_feature
            l2_loss = cnn.l2_loss + cnn_rnn.l2_loss + ncnn.l2_loss
            #l2_loss = ncnn.l2_loss 
            l2_reg_lambda = 0.1

            with tf.variable_scope('whole_final_output'):
                h_drop = tf.nn.dropout(merged_feature, dropout_keep_prob)
                W = tf.get_variable(
                        'W',
                        shape=[merged_feature.shape[1].value, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer()
                        )
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                l2_loss += l2_reg_lambda * tf.nn.l2_loss(W) + l2_reg_lambda * tf.nn.l2_loss(b)
                whole_scores = tf.nn.xw_plus_b(h_drop, W, b, name='whole_scores')
                whole_predictions = tf.argmax(whole_scores, 1, name='whole_predictions')

            with tf.variable_scope('whole_final_loss'):
                entropy_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=whole_scores, labels=input_y)
                whole_loss = tf.reduce_mean(entropy_losses) + l2_loss

            with tf.variable_scope('whole_final_accuracy'):
                correct_predictions = tf.equal(whole_predictions, tf.argmax(input_y, 1))
                whole_accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(whole_loss, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = '../runs/'+model_name + '-' + timestamp
            print('Writing to {}\n'.format(out_dir))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', whole_loss)
            acc_summary = tf.summary.scalar('accuracy', whole_accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

           
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            
            def test(batches):
                #evaluate
                print('Evaluate')
                all_predictions = []
                for batch in batches:  
                    cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list, y = zip(*batch)
                    cnn_batch_x = list(cnn_batch_x)
                    cnn_rnn_batch_x = list(cnn_rnn_batch_x)
                    ncnn_batch_x = list(ncnn_batch_x)
                    sentence_num_list = list(sentence_num_list)
                    y = list(y)
                    feed_dict = {
                            cnn.input_x: cnn_batch_x,
                            cnn_rnn.input_x: cnn_rnn_batch_x,
                            cnn_rnn.real_sentence_num: sentence_num_list,
                            cnn_rnn.dropout_keep_prob: 1.0,
                            cnn_rnn.batch_size: len(y),
                            ncnn.input_x: ncnn_batch_x,
                            dropout_keep_prob: 1.0
                            }

                    batch_predictions = sess.run(whole_predictions, feed_dict)
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

                y_real = np.array(list(map(lambda a: a.index(1), y_test)))
                correct_predictions = sum(all_predictions == y_real)
                print("Total number of test examples: {}".format(len(y_real)))
                print("Accuracy: {:g}".format(correct_predictions/len(y_real)))
                return correct_predictions / len(y_real), y_real, all_predictions 


            batches = data_helpers.batch_iter_whole(x_train_cnn, x_train_ncnn, x_train_cnn_rnn, y_train, 200, 150, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length)
            test_batches = data_helpers.batch_iter_whole(x_test_cnn, x_test_ncnn, x_test_cnn_rnn, y_test, 200, 1, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length, False)
            
            for batch in batches:
                cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list, y = zip(*batch)
                cnn_batch_x = list(cnn_batch_x)
                cnn_rnn_batch_x = list(cnn_rnn_batch_x)
                ncnn_batch_x = list(ncnn_batch_x)
                sentence_num_list = list(sentence_num_list)
                y = list(y)
#                add_noise(ncnn_batch_x, 0.1, 0.3, 0.95, numeric_feature_dim_list)
                feed_dict = {
                        cnn.input_x: cnn_batch_x,
                        cnn_rnn.input_x: cnn_rnn_batch_x,
                        cnn_rnn.batch_size: len(y),
                        cnn_rnn.real_sentence_num: sentence_num_list,
                        cnn_rnn.dropout_keep_prob: 0.6,
                        ncnn.input_x: ncnn_batch_x,
                        input_y: y,
                        dropout_keep_prob: 0.4
                        }
                _, f_step, f_loss, f_accuracy, f_summaries = sess.run([train_op, global_step, whole_loss, whole_accuracy, train_summary_op], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, f_step, f_loss, f_accuracy))
                train_summary_writer.add_summary(f_summaries, f_step) 
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 50 == 0:
                    test_accuracy, y_real, all_predictions = test(test_batches)
                    if test_accuracy > max_accuracy:
                        max_accuracy = test_accuracy 
                        cor_step = current_step
                        data_helpers.save_confusion_matrix(y_real, all_predictions, '../confusion-matrix/'+model_name + '-' + timestamp)
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))
            return max_accuracy, cor_step  

            ##evaluate
            #print('Evaluate')
            #all_predictions = []
            #batches = data_helpers.batch_iter_whole(x_test_cnn, x_test_ncnn, x_test_cnn_rnn, y_test, 128, 1, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length, False)
            #for batch in batches:  
            #    cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list, y = zip(*batch)
            #    cnn_batch_x = list(cnn_batch_x)
            #    cnn_rnn_batch_x = list(cnn_rnn_batch_x)
            #    ncnn_batch_x = list(ncnn_batch_x)
            #    sentence_num_list = list(sentence_num_list)
            #    y = list(y)
            #    feed_dict = {
            #            cnn.input_x: cnn_batch_x,
            #            cnn_rnn.input_x: cnn_rnn_batch_x,
            #            cnn_rnn.real_sentence_num: sentence_num_list,
            #            cnn_rnn.dropout_keep_prob: 1.0,
            #            cnn_rnn.batch_size: len(y),
            #            ncnn.input_x: ncnn_batch_x,
            #            dropout_keep_prob: 1.0
            #            }

            #    batch_predictions = sess.run(whole_predictions, feed_dict)
            #    all_predictions = np.concatenate([all_predictions, batch_predictions])

            #y_test = np.array(list(map(lambda a: a.index(1), y_test)))
            #correct_predictions = sum(all_predictions == y_test)
            #print("Total number of test examples: {}".format(len(y_test)))
            #print("Accuracy: {:g}".format(correct_predictions/len(y_test)))
            #return correct_predictions / len(y_test)


accuracy = 0.0
max_accuracy = 0.0
l = int(sys.argv[1])
for i in range(l):
    train_path = '../data/trainingset' + str(i+1) + '.txt'
    test_path = '../data/testset' + str(i+1) + '.txt'
    x_train_cnn, y_train_cnn = data_helpers.load_text_and_labels(train_path)
    x_train_cnn_rnn, y_train_cnn_rnn = data_helpers.load_sentences_matrix__and_labels(train_path)
    x_train_ncnn , y_train_ncnn, feature_dim_list = data_helpers.load_numeric_matrix_data(train_path)

    assert(y_train_cnn == y_train_cnn_rnn == y_train_ncnn)

    x_test_cnn, y_test_cnn = data_helpers.load_text_and_labels(test_path)
    x_test_cnn_rnn, y_test_cnn_rnn = data_helpers.load_sentences_matrix__and_labels(test_path)
    x_test_ncnn, y_test_ncnn, _ = data_helpers.load_numeric_matrix_data(test_path)

    assert(y_test_cnn == y_test_cnn_rnn == y_test_ncnn)

    a, step= train(x_train_cnn, x_train_cnn_rnn, x_train_ncnn, y_train_cnn, x_test_cnn, x_test_cnn_rnn, x_test_ncnn, y_test_cnn, feature_dim_list)
    print('At step: {}, max accuracy: {}'.format(step, a))
    accuracy += a
    if a > max_accuracy:
        max_accuracy = a

print('Average accuracy: {:g}'.format(accuracy/l))
print('Max accuracy: {:g}'.format(max_accuracy))
 
