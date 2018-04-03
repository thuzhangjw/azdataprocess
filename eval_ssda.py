import tensorflow as tf
import data_helpers
import numpy as np 
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

checkpoint_dir = '/home/zhangjw/anzhentmp/runs/autoencoder-1522746261/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

x_test_numeric, y_test = data_helpers.load_numeric_feature_and_labels('../data/testset.txt')

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name('s_input_x').outputs[0]

        predictions = graph.get_operation_by_name('fine_tune/predictions').outputs[0]
        batches = data_helpers.batch_iter_numeric(x_test_numeric, y_test, 128, 1, False) 

        all_predictions = []
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch, y_batch = list(x_batch), list(y_batch)
            batch_predictions = sess.run(predictions, feed_dict={input_x: x_batch})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

y_test = np.array(list(map(lambda a: a.index(1), y_test)))
correct_predictions = sum(all_predictions == y_test)
print('Total number of test examples: {}'.format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/len(y_test)))

