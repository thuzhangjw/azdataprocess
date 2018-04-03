import tensorflow as tf
from autoencoder import StackedAutoencoderClassifier
from cnn import CNNLayer
import pandas as pd
from gensim.models import Word2Vec
import os
import data_helpers
import datetime
import time


text_features_train, numeric_features_train, labels_train = data_helpers.load_all_data('../data/trainingset.txt')

text_features_test, numeric_features_test, labels_test = data_helpers.load_all_data('../data/testset.txt')

max_sentence_length = max([len(x.split(' ')) for x in (text_features_train + text_features_test)])
numeric_feature_num = len(numeric_features_train[0])

init_words_embedded_model = Word2Vec.load('../data/word2vec.model')
num_classes = len(labels_train[0])
l2_reg_lambda = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True 
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        input_y = tf.placeholder(tf.float32, [None, num_classes], name='final_input_y')
        cnn = CNNLayer(
                sequence_length = max_sentence_length,
                filter_sizes=[2, 3, 4, 5],
                num_filters=64,
                init_words_embedded_model=init_words_embedded_model,
                num_classes=num_classes,
                l2_reg_lambda=0
                )

        sa = StackedAutoencoderClassifier(
                n_input=numeric_feature_num,
                n_hidden_list=[180, 100, 50],
                num_classes=num_classes,
                l2_reg_lambda=0
                )

        merged_feature = tf.concat([cnn.h_drop, sa.final_features], 1)
        with tf.variable_scope('final_output'):
            W = tf.get_variable(
                    'W',
                    shape=[merged_feature.shape[1].value, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.get_variable(
                    'b',
                    shape=[num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )

            l2_loss = tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(merged_feature, W, b, name='scores')
            predictions = tf.argmax(scores, 1, name='predictions')

        with tf.variable_scope('final_loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=input_y)
            loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.variable_scope('final_accuracy'):
            correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = '../run/final_model-' + timestamp
        print('Writing to {}\n'.format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # pre train
        batches = data_helpers.batch_iter_numeric(numeric_features_train, labels_train, 128, 40)
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


        # train the whole model
        batches = data_helpers.batch_iter(text_features_train, numeric_features_train, labels_train, 100, max_sentence_length)
        for batch in batches:
            text_batch, numeric_and_label_batch = zip(*batch)
            numeric_batch, label_batch = zip(*numeric_and_label_batch)
            text_batch, numeric_batch, label_batch = list(text_batch), list(numeric_batch), list(label_batch)
            feed_dict = {
                cnn.input_x: text_batch,
                cnn.dropout_keep_prob: 0.5, 
                sa.input_x: numeric_batch,
                input_y: label_batch 
                    }

            _, t_step, t_loss, t_accuracy = sess.run([train_op, global_step, loss, accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, t_step, t_loss, t_accuracy))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        # write fine tuned word embedding to file
        new_wordvec_path = '../data/new_wordvec_wholemodel.txt'
        print('Write fine tuned word vector to file {}'.format(new_wordvec_path))
        with open(new_wordvec_path, 'w') as f:
            f.write(str(cnn.embedded_vocabulary.shape[0]) + ' ' + str(cnn.embedded_vocabulary.shape[1]))
            idx = 0
            for val in sess.run(cnn.embedded_vocabulary):
                if idx == 0:
                    f.write('\n' + data_helpers.wordvec2str('unknown', val))
                else:
                    f.write('\n' + data_helpers.wordvec2str(cnn.keys[idx-1], val))
                idx += 1

