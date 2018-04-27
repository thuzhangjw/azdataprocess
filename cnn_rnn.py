import tensorflow as tf
import pandas as pd
from gensim.models import Word2Vec 


class CNN_RNN(object):

    def __init__(self, sentence_num, sentence_length, filter_sizes, num_filters, init_words_embedded_model, rnn_hidden_dim, num_classes, l2_reg_lambda=0.1, use_static=True):

        self.use_static = use_static
        self.vocabulary_index_map, self.embedded_vocabulary = self.load_init_embedded_vocabulary(init_words_embedded_model)
        embedding_size = init_words_embedded_model.vector_size
        self.input_x = tf.placeholder(tf.string, [None, sentence_num, sentence_length], name='cr_input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='cr_input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='cr_dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [], name='cr_batch_size')
        self.real_sentence_num = tf.placeholder(tf.int32, [None], name='real_sentence_num')
    
        if self.use_static:
            self.static_embedded_vocabulary = tf.Variable(self.embedded_vocabulary, trainable=False)
        sentences_embedded = []
        for i in range(sentence_num):
            sentence = self.input_x[:, i, :]
            vocab_indices = self.vocabulary_index_map.lookup(sentence)
            embedded_chars = tf.nn.embedding_lookup(self.embedded_vocabulary, vocab_indices)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            if self.use_static:
                static_embedded_chars = tf.nn.embedding_lookup(self.static_embedded_vocabulary, vocab_indices)
                static_embedded_chars_expanded = tf.expand_dims(static_embedded_chars, -1)
                embedded_chars_expanded = tf.concat([embedded_chars_expanded, static_embedded_chars_expanded], -1)
            sentences_embedded.append(embedded_chars_expanded)

        l2_loss = tf.constant(0.0)
        sentences_pooled_outputs = [[] for _ in range(sentence_num)]
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('cnn_rnn_convolution-maxpool-%s' % filter_size):
                filter_shape = [filter_size,  embedding_size, [1,2][self.use_static], num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                for j, se in enumerate(sentences_embedded):
                    conv = tf.nn.conv2d(se, W, strides=[1,1,1,1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, sentence_length-filter_size+1, 1, 1],
                            strides=[1,1,1,1],
                            padding='VALID'
                            )
                    sentences_pooled_outputs[j].append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        sentences_flat = []
        for sentence_pooled in sentences_pooled_outputs:
            h_pool = tf.concat(sentence_pooled, axis=3)
            h_pool_flat = tf.reshape(h_pool, [-1, 1, num_filters_total])
            sentences_flat.append(h_pool_flat)

        rnn_inputs = tf.concat(sentences_flat, axis=1)
        rnn_cell = self.unit_rnn(rnn_hidden_dim)
        initial_state = rnn_cell.zero_state(self.batch_size, tf.float32)
        rnn_outputs, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=rnn_inputs, dtype=tf.float32, initial_state=initial_state, sequence_length=self.real_sentence_num)
        self.rnn_last_output = tf.concat([rnn_outputs[:, -1, :], rnn_state[0]], axis=1)
        #self.rnn_last_output = tf.concat([tf.reduce_mean(rnn_outputs, 1), rnn_state[0]], axis=1)
        #self.rnn_last_output = tf.reshape(rnn_outputs, [self.batch_size, rnn_outputs.shape[1].value*rnn_outputs.shape[2].value])

        with tf.variable_scope('cnn_rnn_final_feature'):
            W = tf.get_variable(
                    'W',
                    shape=[self.rnn_last_output.shape[1].value, self.rnn_last_output.shape[1].value],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.Variable(tf.constant(0.1, shape=[self.rnn_last_output.shape[1].value]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.l2_loss = l2_reg_lambda * l2_loss 
            self.final_feature = tf.nn.xw_plus_b(self.rnn_last_output, W, b, name='final_feature')


        with tf.variable_scope('rnn_outputs'):
            self.h_drop = tf.nn.dropout(self.final_feature, self.dropout_keep_prob)
            W = tf.get_variable(
                    'W',
                    shape=[self.rnn_last_output.shape[1].value, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss = tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
            
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * l2_loss

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='cr_accuracy')


    def unit_rnn(self, hidden_dim):
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_dim)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        return cell

    
    def load_init_embedded_vocabulary(self, init_words_embedded_model):
        wv = init_words_embedded_model.wv
        vector_size = wv.vector_size 
        
        embedded_words_list = []
        self.keys = []
        self.vals = []

        embedded_words_list.append([0]*vector_size)
        
        for i, w in enumerate(wv.vocab):
            embedded_words_list.append(list(wv[w]))
            # vocabulary_index_map[w] = i + 1
            self.keys.append(w)
            self.vals.append(i+1)

        embedded_vocabulary = tf.Variable(embedded_words_list, name='Vocabulary')
        vocabulary_index_map = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(self.keys, self.vals), 0, name='vocabulary_index_map')
        
        return vocabulary_index_map, embedded_vocabulary 


