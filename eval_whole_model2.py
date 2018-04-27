import tensorflow as tf
import data_helpers
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

checkpoint_dir = '/home/zhangjw/anzhentmp/run/final_model2-1523341560/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

train_path = '../data/testset1.txt'
test_path = '../data/trainingset1.txt'

x_train_cnn, y_train_cnn = data_helpers.load_text_and_labels(train_path)
x_train_cnn_rnn, y_train_cnn_rnn = data_helpers.load_sentences_matrix__and_labels(train_path) 
x_test_cnn, y_test_cnn = data_helpers.load_text_and_labels(test_path)
x_test_cnn_rnn, y_test_cnn_rnn = data_helpers.load_sentences_matrix__and_labels(test_path)
x_test_ncnn, y_test_ncnn = data_helpers.load_numeric_matrix_data(test_path)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
 
        cnn_input_x = graph.get_operation_by_name('input_x').outputs[0]
        cnn_rnn_input_x = graph.get_operation_by_name('cr_input_x').outputs[0]
        cnn_rnn_real_sentence_num = graph.get_operation_by_name('real_sentence_num').outputs[0]
        cnn_rnn_dropout_keep_prob = graph.get_operation_by_name('cr_dropout_keep_prob').outputs[0]
        cnn_rnn_batch_size = graph.get_operation_by_name('cr_batch_size').outputs[0]
        ncnn_input_x = graph.get_operation_by_name('nc_input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('whole_dropout_keep_prob').outputs[0]
        whole_predictions = graph.get_operation_by_name('whole_final_output/whole_predictions').outputs[0]

        cnn_max_sentence_length = max([len(x.split(' ')) for x in (x_train_cnn + x_test_cnn)])
        cnn_rnn_max_sentence_num = max([len(a) for a in (x_train_cnn_rnn + x_test_cnn_rnn)])
        cnn_rnn_max_sentence_length = data_helpers.get_cnn_rnn_max_sentence_length(x_train_cnn_rnn, x_test_cnn_rnn)
 
        sess.run(tf.tables_initializer())

        all_predictions = []
        batches = data_helpers.batch_iter_whole(x_test_cnn, x_test_ncnn, x_test_cnn_rnn, y_test_cnn, 128, 1, cnn_max_sentence_length, cnn_rnn_max_sentence_num, cnn_rnn_max_sentence_length, False)
        for batch in batches:  
            cnn_batch_x, cnn_rnn_batch_x, ncnn_batch_x, sentence_num_list, y = zip(*batch)
            cnn_batch_x = list(cnn_batch_x)
            cnn_rnn_batch_x = list(cnn_rnn_batch_x)
            ncnn_batch_x = list(ncnn_batch_x)
            sentence_num_list = list(sentence_num_list)
            y = list(y)
            feed_dict = {
                    cnn_input_x: cnn_batch_x,
                    cnn_rnn_input_x: cnn_rnn_batch_x,
                    cnn_rnn_real_sentence_num: sentence_num_list,
                    cnn_rnn_dropout_keep_prob: 1.0,
                    cnn_rnn_batch_size: len(y),
                    ncnn_input_x: ncnn_batch_x,
                    dropout_keep_prob: 1.0
                    }

            batch_predictions = sess.run(whole_predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        y_test = np.array(list(map(lambda a: a.index(1), y_test_cnn)))
        correct_predictions = sum(all_predictions == y_test)
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/len(y_test)))


