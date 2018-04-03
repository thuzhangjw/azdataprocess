import tensorflow as tf


class Autoencoder():
    def __init__(self, layer_index, n_input, n_hidden, l2_lambda=0.1, sparse=True, sparse_ratio=0.1, sparse_beta=3, denoising=True, denoise_ratio=0.1):
        
        with tf.variable_scope('layer-%s' % layer_index):
            self.n_input = n_input 
            self.denoise_ratio = denoise_ratio 
            self.input_x = tf.placeholder(tf.float32, [None, n_input], name='input_x') 
            self.hidden_weight = tf.get_variable(
                    'W',
                    shape=[n_input, n_hidden],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            self.hidden_bias =tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b')
            out_weight = tf.get_variable(
                    'out_w',
                    shape=[n_hidden, n_input],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            out_bias = tf.Variable(tf.constant(0.1, shape=[n_input]))

            hidden = self.comput_hidden(self.input_x, self.hidden_weight, self.hidden_bias, denoising)

            out = tf.sigmoid(tf.nn.xw_plus_b(hidden, out_weight, out_bias))

            self.loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(self.input_x, out), 2), axis=1))
            self.loss += l2_lambda * (tf.nn.l2_loss(self.hidden_weight) + tf.nn.l2_loss(out_weight) + tf.nn.l2_loss(self.hidden_bias) + tf.nn.l2_loss(out_bias))

            if sparse:
                p_hats = tf.reduce_mean(hidden, axis=0)
                self.loss += sparse_beta * self.kl_divergence(sparse_ratio, p_hats)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)

    
    def kl_divergence(self, p, p_hats):
        return tf.reduce_sum(p * tf.log(p/p_hats) + (1-p) * tf.log((1-p)/(1-p_hats)))
        
    
    def add_noise(self, x):
        mask = tf.keras.backend.random_binomial([self.n_input], p=(1-self.denoise_ratio))
        return x*mask 

    def comput_hidden(self, x, W, b, denoising):
        if denoising:
            x = self.add_noise(x)
        return tf.sigmoid(tf.nn.xw_plus_b(x, W, b))


class StackedAutoencoderClassifier():
    def __init__(self, n_input, n_hidden_list, num_classes, l2_reg_lambda=0.1):

        num_hidden = len(n_hidden_list)
        self.autoencoders =[Autoencoder(1, n_input, n_hidden_list[0])]
        for i in range(len(n_hidden_list)-1):
            self.autoencoders.append(Autoencoder(i+2, n_hidden_list[i], n_hidden_list[i+1]))
        
        self.input_x = tf.placeholder(tf.float32, [None, n_input], name='s_input_x')
        self.input_y =tf.placeholder(tf.float32, [None, num_classes], name='input_y')

        # Fine tune
        with tf.variable_scope('fine_tune'):
            self.final_features = self.input_x
            for encoder in self.autoencoders:
                self.final_features = encoder.comput_hidden(self.final_features, encoder.hidden_weight, encoder.hidden_bias, False)
            W = tf.get_variable(
                    'W',
                    shape=[n_hidden_list[num_hidden-1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                    )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss = tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.final_features, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
        self.loss =tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)


