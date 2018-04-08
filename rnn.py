import tensorflow as tf


class RNNLayer(object):

    def __init__(self, layer_num, sequence_length, hidden_dim, num_classes, init_words_embedded_model, l2_reg_lambda=0.1):

        self.vocabulary_index_map, self.embedded_vocabulary = self.load_init_embedded_vocabulary(init_words_embedded_model)
        self.input_x = tf.placeholder(tf.string, [None, sequence_length], name='r_input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='r_input_y')
        self.drop_out_prob = tf.placeholder(tf.float32, name='r_dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [], name='r_batch_size')

    
        vocab_indices = self.vocabulary_index_map.lookup(self.input_x)
        embedded_inputs = tf.nn.embedding_lookup(self.embedded_vocabulary, vocab_indices)


        # multi rnn layer
        #cells = [self.unit_lstm(hidden_dim) for _ in range(layer_num)]
        #rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
        rnn_cell = self.unit_lstm(hidden_dim)
        initial_state = rnn_cell.zero_state(self.batch_size, tf.float32)
        
        outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedded_inputs, dtype=tf.float32, initial_state=initial_state)
        last = outputs[:, -1, :]
        self.fc = last
        #self.fc = tf.nn.dropout(last, self.drop_out_prob)
        #self.fc = tf.nn.relu(self.fc)
        
        W = tf.get_variable(
                'r_W',
                shape=[hidden_dim, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
                )
        b = tf.get_variable(
                'r_b',
                shape=[num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
                )

        l2_loss = tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.fc, W, b, name='r_scores')
        self.predictions = tf.argmax(self.scores, 1, name='r-predictions')

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * l2_loss

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='r_accuracy')
        

    def unit_lstm(self, hidden_dim):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.drop_out_prob)
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

        embedded_vocabulary = tf.Variable(embedded_words_list, name='r_Vocabulary')
        vocabulary_index_map = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(self.keys, self.vals), 0, name='vocabulary_index_map')
        
        return vocabulary_index_map, embedded_vocabulary 
