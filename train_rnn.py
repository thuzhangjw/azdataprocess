import tensorflow as tf
import pandas as pd
from rnn import RNNLayer
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
    session_conf.gpu_options.allow_growth = True 
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        rnn = RNNLayer(
                layer_num=2,
                sequence_length=max_sentence_length,
                hidden_dim=256,
                num_classes=len(y_train[0]),
                init_words_embedded_model=init_words_embedded_model,
                l2_reg_lambda=0.1
                )
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(rnn.loss, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = '../runs/rnn-' + timestamp
        print('Writing to {}\n'.format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        batches = data_helpers.batch_iter_text(x_train_text, y_train, 128, 200, max_sentence_length)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch, y_batch = list(x_batch), list(y_batch)

            feed_dict ={
                    rnn.input_x: x_batch,
                    rnn.input_y: y_batch,
                    rnn.drop_out_prob: 0.5,
                    rnn.batch_size: len(y_batch)
                    }

            _, t_step, t_loss, t_accuracy = sess.run([train_op, global_step, rnn.loss, rnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, t_step, t_loss, t_accuracy))
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 500 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
         
        # write fine tuned word embedding to file
        new_wordvec_path = '../data/new_wordvec_rnnmodel.txt'
        print('Write fine tuned word vector to file {}'.format(new_wordvec_path))
        with open(new_wordvec_path, 'w') as f:
            f.write(str(rnn.embedded_vocabulary.shape[0]) + ' ' + str(rnn.embedded_vocabulary.shape[1]))
            idx = 0
            for val in sess.run(rnn.embedded_vocabulary):
                if idx == 0:
                    f.write('\n' + data_helpers.wordvec2str('unknown', val))
                else:
                    f.write('\n' + data_helpers.wordvec2str(rnn.keys[idx-1], val))
                idx += 1

