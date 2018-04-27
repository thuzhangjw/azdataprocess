import tensorflow as tf
import pandas as pd
import data_helpers
import os
import time
import numpy as np
import datetime
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name='dbn'

class RBM(object):
    def __init__(self, layer_index, n_input, n_hidden, cd_k=1, rbm_lr=0.1):
        self.n_v = n_input
        self.n_h = n_hidden 
        with tf.variable_scope('layer-%s' % layer_index):
            self.input_x = tf.placeholder(tf.float32, [None, n_input], name='input_x')
            self.weight = tf.Variable(tf.truncated_normal(shape=[n_input, n_hidden], stddev=0.1), name='hidden_weight')
            self.bv = tf.Variable(tf.constant(0.0,shape=[n_input]),name='bv')
            self.bh = tf.Variable(tf.constant(0.0,shape=[n_hidden]),name='bh')

            v0 = self.input_x 
            h0, s_h0 = self.transform(v0)
            vk = self.reconstruction(s_h0)
            for k in range(cd_k-1):
                _, s_hk = self.transform(vk)
                vk = self.reconstruction(s_hk)

            hk, _ = self.transform(vk)
            positive=tf.matmul(tf.expand_dims(v0, -1), tf.expand_dims(h0, 1))
            negative=tf.matmul(tf.expand_dims(vk, -1), tf.expand_dims(hk, 1))
            self.w_upd8 = self.weight.assign_add(tf.multiply(rbm_lr, tf.reduce_mean(tf.subtract(positive, negative), 0)))
            self.bv_upd8 = self.bv.assign_add(tf.multiply(rbm_lr, tf.reduce_mean(tf.subtract(v0, vk), 0)))
            self.bh_upd8 = self.bh.assign_add(tf.multiply(rbm_lr, tf.reduce_mean(tf.subtract(h0, hk), 0)))
            self.train_batch = [self.w_upd8, self.bv_upd8, self.bh_upd8]

            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(v0, vk))))

    
    def transform(self, v):
        z = tf.add(tf.matmul(v, self.weight), self.bh)
        prob_h=tf.nn.sigmoid(z,name='prob_h') 
        state_h= tf.to_float(tf.random_uniform([tf.shape(v)[0], self.n_h])<prob_h, name='state_h') # sample
        return prob_h,state_h


    def reconstruction(self, h):
        z = tf.add(tf.matmul(h, tf.transpose(self.weight)), self.bv)
        prob_v=tf.nn.sigmoid(z, name='prob_v')
        return prob_v 


class DBN(object):
    def __init__(self, n_input, n_hidden_list, num_classes, l2_reg_lambda=0.1, cd_k=1, rbm_lr=0.1):
        self.rbms = [RBM(1, n_input, n_hidden_list[0], cd_k, rbm_lr)]
        for i in range(len(n_hidden_list)-1):
            self.rbms.append(RBM(i+2, n_hidden_list[i], n_hidden_list[i+1], cd_k, rbm_lr))

        self.input_x = tf.placeholder(tf.float32, [None, n_input], name='rbm_input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='rbm_input_y')

        with tf.variable_scope('DBN_fine_tune'):
            self.final_features = self.input_x
            for rbm in self.rbms:
                self.final_features, _ = rbm.transform(self.final_features)
            W = tf.get_variable(
                    'W',
                    shape=[n_hidden_list[-1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss = tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.final_features, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
        self.loss =tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)


def train(train_x, train_y, test_x, test_y, feature_num):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            dbn = DBN(
                    n_input=feature_num,
                    n_hidden_list=[200, 100, 60],
                    num_classes=len(train_y[0]),
                    l2_reg_lambda=0.1,
                    cd_k=1,
                    rbm_lr=0.1
                    )
            timestamp = str(int(time.time()))
            out_dir = '../runs/dbn-' + timestamp
            print('Writing to {}\n'.format(out_dir))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    
            sess.run(tf.global_variables_initializer())
 
            print('pre training')
            batches = data_helpers.batch_iter_numeric(train_x, train_y, 256, 30)
            for i in range(len(dbn.rbms)):
                for batch in batches:
                    x_batch, _ = zip(*batch)
                    x_batch = list(x_batch)
                    for j in range(i):
                        x_batch, _ = dbn.rbms[j].transform(x_batch)
                    if i > 0:
                        x_batch = sess.run(x_batch)
                    _, loss =  sess.run([dbn.rbms[i].train_batch, dbn.rbms[i].loss], feed_dict={dbn.rbms[i].input_x: x_batch})
                    print('{} Layer: loss {:g}'.format(i, loss))

            print('\nfine tune')
            batches = data_helpers.batch_iter_numeric(train_x, train_y, 256, 80)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, y_batch = list(x_batch), list(y_batch)
                feed_dict = {
                        dbn.input_x: x_batch,
                        dbn.input_y: y_batch 
                        }
                _, loss, accuracy = sess.run([dbn.train_op, dbn.loss, dbn.accuracy], feed_dict)
                            
                time_str = datetime.datetime.now().isoformat()
                print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))
             
             # evaluate
            print('evaluate')
            batches = data_helpers.batch_iter_numeric(test_x, test_y, 128, 1, False) 
            all_predictions = []
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch, y_batch = list(x_batch), list(y_batch)
                batch_predictions = sess.run(dbn.predictions, feed_dict={dbn.input_x: x_batch})
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

