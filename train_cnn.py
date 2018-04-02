import tensorflow as tf
import pandas as pd
import re
from cnn import CNNLayer  
from gensim.models import Word2Vec 
import time
import os
import datetime
import data_helpers 


x_train_text, y_train = data_helpers.load_text_and_labels('../data/trainingset.txt')
x_test_text, y_test = data_helpers.load_text_and_labels('../data/testset.txt')

max_sentence_length = max([len(x.split(" ")) for x in (x_train_text + x_test_text)])

init_words_embedded_model = Word2Vec.load('../data/word2vec.model')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNNLayer(
                sequence_length = max_sentence_length, 
                filter_sizes = [2, 3, 4, 5],
                num_filters = 128, 
                init_words_embedded_model = init_words_embedded_model,
                num_classes = len(y_train[0]),
                l2_reg_lambda = 0.1
                )

        # Define Training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = '../runs/' + timestamp 
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        # cnn.vocabulary_index_map.init.run() 
        sess.run(tf.tables_initializer())

        def train_step(x_batch, y_batch):
           
            feed_dict={
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 0.5
                    }
            _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def test_step(x_batch, y_batch):

            feed_dict={
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                    }

            step, summaries, loss, accuracy = sess.run(
                    [global_step, test_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict
                    )
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            
        
        # Generate batches
        batches =  data_helpers.batch_iter_text(x_train_text, y_train, 64, 150, max_sentence_length)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch, y_batch = list(x_batch), list(y_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                
       
        # write fine tuned word embedding to file
        new_wordvec_path = '../data/new_wordvec.txt'
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

