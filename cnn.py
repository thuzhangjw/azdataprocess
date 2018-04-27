import tensorflow as tf
import pandas as pd
from gensim.models import Word2Vec 

class CNNLayer(object):
    def __init__(self, sequence_length, filter_sizes, num_filters, init_words_embedded_model, num_classes, l2_reg_lambda=0.1, use_static=False):
        
        self.use_static = use_static 
        self.vocabulary_index_map, self.embedded_vocabulary = self.load_init_embedded_vocabulary(init_words_embedded_model)
        embedding_size = init_words_embedded_model.vector_size
        self.input_x = tf.placeholder(tf.string, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        if use_static:
            self.static_embedded_vocabulary = tf.Variable(self.embedded_vocabulary, trainable=False)
    
        l2_loss = tf.constant(0.0)

        # Embedding Layer
        with tf.name_scope('embedding'):
            vocab_indices = self.vocabulary_index_map.lookup(self.input_x)
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_vocabulary, vocab_indices)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            if use_static:
                static_embedded_chars = tf.nn.embedding_lookup(self.static_embedded_vocabulary, vocab_indices)
                static_embedded_chars_expanded = tf.expand_dims(static_embedded_chars, -1)
                self.embedded_chars_expanded = tf.concat([self.embedded_chars_expanded, static_embedded_chars_expanded], -1)

        # Creating convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i , filter_size in enumerate(filter_sizes):
            with tf.name_scope('convolution-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, self.embedded_chars_expanded.shape[-1].value, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv= tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                        h, 
                        ksize=[1, sequence_length-filter_size+1, 1, 1],
                        strides=[1,1,1,1],
                        padding='VALID',
                        name='pool')
                pooled_outputs.append(pooled)
        
        # Combine all thr pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.variable_scope('final_feature'):
            W = tf.get_variable(
                    'W',
                    shape=[num_filters_total, num_filters_total],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.l2_loss = l2_reg_lambda * l2_loss 
            self.final_feature = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name='final_feature')

        # Final scores and predictions
        with tf.name_scope('output'):
            self.h_drop = tf.nn.dropout(self.final_feature, self.dropout_keep_prob)
            W = tf.get_variable(
                    'W', 
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss 

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
    

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


