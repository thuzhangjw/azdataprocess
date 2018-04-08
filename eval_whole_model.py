import tensorflow as tf
import data_helpers
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

checkpoint_dir = '/home/zhangjw/anzhentmp/run/final_model-1522832662/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

text_features_test, numeric_features_test, labels_test = data_helpers.load_all_data('../data/testset.txt')

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        sa_input_x = graph.get_operation_by_name('s_input_x').outputs[0]
        cnn_input_x = graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

        predictions = graph.get_operation_by_name('final_output/predictions').outputs[0]
        max_sentence_length = cnn_input_x.shape[1]

        sess.run(tf.tables_initializer())
        batches = data_helpers.batch_iter(text_features_test, numeric_features_test, labels_test, 128, 1, max_sentence_length, shuffle=False)

        all_predictions = []
        for batch in batches:
            text_batch, numeric_and_label_batch = zip(*batch)
            numeric_batch, label_batch = zip(*numeric_and_label_batch)
            text_batch, numeric_batch, label_batch = list(text_batch), list(numeric_batch), list(label_batch)

            feed_dict = {
                    sa_input_x: numeric_batch,
                    cnn_input_x: text_batch,
                    dropout_keep_prob: 1.0
                    }
            batch_predictions = sess.run(predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

y_test = np.array(list(map(lambda a: a.index(1), labels_test)))
correct_predictions = sum(all_predictions == y_test)
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/len(y_test)))
        
