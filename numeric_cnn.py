import tensorflow as tf

class NumericCNN(object):

    def __init__(self, feature_num, feature_dim, filter_sizes, num_filters, num_classes, l2_reg_lambda=0.1):

        self.input_x = tf.placeholder(tf.float32, [None, feature_num, feature_dim], name='nc_input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='nc_input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='nc_dropout_keep_prob')

        input_x_expanded = tf.expand_dims(self.input_x, -1)
        
        l2_loss = tf.constant(0.0)
        pooled_outputs =[]
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('nc_convolution-%s' % filter_size):
                filter_shape = [filter_size, feature_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b'))
                conv = tf.nn.conv2d(input_x_expanded, W, strides=[1,1,1,1], padding='VALID', name='conv')

#                l2_loss += tf.nn.l2_loss(W)
#                l2_loss += tf.nn.l2_loss(b)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, feature_num-filter_size+1, 1, 1],
                        strides=[1,1,1,1],
                        padding='VALID',
                        name='pool')
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        
        with tf.variable_scope('ncnn_final_feature'):
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


        #dropout
        self.h_drop = tf.nn.dropout(self.final_feature, self.dropout_keep_prob)

        with tf.variable_scope('nc_output'):
            W = tf.get_variable(
                    'W',
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss = tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss 
        
        with tf.variable_scope('nc_accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


